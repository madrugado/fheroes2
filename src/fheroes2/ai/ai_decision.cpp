/***************************************************************************
 *   fheroes2: https://github.com/ihhub/fheroes2                           *
 *   Copyright (C) 2026                                                    *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include "ai_decision.h"

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ai_planner.h"
#include "army.h"
#include "artifact.h"
#include "battle_server.h"
#include "castle.h"
#include "color.h"
#include "game.h"
#include "heroes.h"
#include "kingdom.h"
#include "logging.h"
#include "maps.h"
#include "maps_tiles.h"
#include "monster.h"
#include "payment.h"
#include "rand.h"
#include "resource.h"
#include "skill.h"
#include "world.h"

namespace
{
    bool prepareDecisionChannel()
    {
        static bool initialized = false;
        static bool enabled = false;

        if ( !initialized ) {
            initialized = true;
            const char * value = std::getenv( "FHEROES2_STRATEGY_SERVER" );
            enabled = ( value != nullptr && *value != '\0' );

#ifdef SIGPIPE
            if ( enabled ) {
                // The agent may die at any moment: a write to its closed pipe must fail (the
                // channel then breaks on the next read and the built-in AI takes over) instead
                // of killing the game with SIGPIPE.
                std::signal( SIGPIPE, SIG_IGN );
            }
#endif
        }

        return enabled;
    }

    // The protocol channel is broken (the agent is gone or misbehaves): fall back to the
    // built-in AI for the rest of the session.
    bool channelBroken = false;

    bool isChannelBroken()
    {
        return channelBroken;
    }

    void markChannelBroken()
    {
        if ( !channelBroken ) {
            channelBroken = true;
            ERROR_LOG( "Strategy decision channel is broken: falling back to the built-in AI." )
        }
    }

    void writeFunds( std::ostringstream & out, const char * key, const Funds & funds )
    {
        out << ",\"" << key << "\":[" << funds.wood << ',' << funds.mercury << ',' << funds.ore << ',' << funds.sulfur << ',' << funds.crystal << ',' << funds.gems
            << ',' << funds.gold << ']';
    }

    // Waits for the agent's reply to a choice request: a line with the expected operation or a
    // "skip". Unknown lines are ignored. Returns false (and breaks the channel) if the agent is gone.
    bool readReply( const char * expectedOp, std::string & line )
    {
        while ( std::getline( std::cin, line ) ) {
            if ( line.find( expectedOp ) != std::string::npos || line.find( "\"skip\"" ) != std::string::npos ) {
                return true;
            }
        }

        markChannelBroken();
        return false;
    }

    int64_t extractInt( const std::string & line, const char * key, const int64_t defaultValue )
    {
        const std::string needle = "\"" + std::string( key ) + "\":";
        const size_t pos = line.find( needle );
        if ( pos == std::string::npos ) {
            return defaultValue;
        }

        return std::strtoll( line.c_str() + pos + needle.size(), nullptr, 10 );
    }
}

bool AIDecision::isEnabled()
{
    return prepareDecisionChannel() && !isChannelBroken();
}

void AIDecision::writeKingdomStats( std::ostringstream & out, const PlayerColor color )
{
    const Kingdom & kingdom = world.GetKingdom( color );

    double strength = 0;
    for ( const Heroes * hero : kingdom.GetHeroes() ) {
        strength += hero->GetArmy().GetStrength();
    }
    for ( const Castle * castle : kingdom.GetCastles() ) {
        strength += castle->GetArmy().GetStrength();
    }

    out << ",\"k\":" << kingdom.GetCastles().size() << ",\"h\":" << kingdom.GetHeroes().size() << ",\"str\":" << static_cast<int64_t>( strength )
        << ",\"g\":" << kingdom.GetFunds().gold;

    const Heroes * strongest = nullptr;
    for ( const Heroes * hero : kingdom.GetHeroes() ) {
        if ( strongest == nullptr || hero->GetArmy().GetStrength() > strongest->GetArmy().GetStrength() ) {
            strongest = hero;
        }
    }
    if ( strongest != nullptr ) {
        out << ",\"top\":{\"hid\":" << strongest->GetID() << ",\"str\":" << static_cast<int64_t>( strongest->GetArmy().GetStrength() )
            << ",\"hero\":\"" << Battle::EncodeCommander( strongest->GetArmy() ) << "\"}";
    }
}

namespace
{
    // FHEROES2_REPORT_DAYS: the days whose first AI turn is preceded by a "day_report".
    bool isReportDay( const uint32_t day )
    {
        static const std::vector<uint32_t> days = [] {
            std::vector<uint32_t> result;
            const char * value = std::getenv( "FHEROES2_REPORT_DAYS" );
            if ( value != nullptr ) {
                std::istringstream in( value );
                std::string item;
                while ( std::getline( in, item, ',' ) ) {
                    const long parsed = std::strtol( item.c_str(), nullptr, 10 );
                    if ( parsed > 0 ) {
                        result.push_back( static_cast<uint32_t>( parsed ) );
                    }
                }
            }
            return result;
        }();
        return std::find( days.begin(), days.end(), day ) != days.end();
    }

    void sendDayReport()
    {
        static uint32_t lastReportedDay = 0;
        const uint32_t day = world.CountDay();
        if ( day == lastReportedDay || !isReportDay( day ) ) {
            return;
        }
        lastReportedDay = day;

        std::ostringstream out;
        out << "{\"ev\":\"day_report\",\"t\":" << day << ",\"results\":[";
        bool first = true;
        for ( const PlayerColor color : PlayerColorsVector( Color::allPlayerColors() ) ) {
            if ( !world.GetKingdom( color ).isPlay() ) {
                continue;
            }
            if ( !first ) {
                out << ',';
            }
            first = false;
            out << "{\"c\":\"" << Color::String( color ) << '"';
            AIDecision::writeKingdomStats( out, color );
            out << '}';
        }
        out << "]}\n";
        std::cout << out.str();
        std::cout.flush();
    }
}

namespace
{
    // The lower bound of the army size word a player sees for an enemy stack ("few" 1-4, "several"
    // 5-9, ..., "legion" 1000+; Army::SizeString).
    uint32_t visibleCountBand( const uint32_t count )
    {
        uint32_t band = 1;
        for ( const uint32_t bound : { 5U, 10U, 20U, 50U, 100U, 250U, 500U, 1000U } ) {
            if ( count >= bound ) {
                band = bound;
            }
        }
        return band;
    }

    // How much of an army a player sees: nothing, monster types only, types with the size word,
    // or exact counts (see drawMiniMonsters in army_ui_helper.cpp).
    enum class ArmyView
    {
        UNKNOWN,
        TYPES,
        BANDS,
        EXACT
    };

    // [monster, count, strength of one creature, level, speed, shooter, flyer]: what the monster's
    // info dialog shows about a creature type.
    void writeMonster( std::ostringstream & out, const Monster & monster, const uint32_t count )
    {
        out << '[' << monster.GetID() << ',' << count << ',' << static_cast<int64_t>( monster.GetMonsterStrength() * 100 ) / 100.0 << ','
            << monster.GetMonsterLevel() << ',' << monster.GetSpeed() << ',' << ( monster.isArchers() ? 1 : 0 ) << ',' << ( monster.isFlying() ? 1 : 0 ) << ']';
    }

    // Writes ,"army":[stack,...] (writeMonster; count 0 = not shown) and returns the strength
    // estimated from what is shown (types only: one monster per stack).
    double writeArmy( std::ostringstream & out, const Army & army, const ArmyView view )
    {
        out << ",\"army\":[";
        double estimate = 0;
        bool first = true;
        if ( view != ArmyView::UNKNOWN ) {
            for ( size_t slot = 0; slot < army.Size(); ++slot ) {
                const Troop * troop = army.GetTroop( slot );
                if ( troop == nullptr || !troop->isValid() ) {
                    continue;
                }
                uint32_t shown = 0;
                if ( view == ArmyView::EXACT ) {
                    shown = troop->GetCount();
                }
                else if ( view == ArmyView::BANDS ) {
                    shown = visibleCountBand( troop->GetCount() );
                }
                estimate += troop->GetMonsterStrength() * std::max( shown, 1U );
                if ( !first ) {
                    out << ',';
                }
                first = false;
                writeMonster( out, *troop, shown );
            }
        }
        out << ']';
        return estimate;
    }

    // Enemy heroes as a human player of this kingdom would see them on the adventure map (the quick
    // info of dialog_quickinfo.cpp): only heroes on tiles outside the fog; the army as monster
    // types with the size word; with full information (the Identify Hero spell, the Crystal Ball
    // view) also the exact counts, primary skills, level, spell and move points, morale and luck.
    // The spell book is never visible. "est" is the army strength estimated from what is shown.
    void writeVisibleRivals( std::ostringstream & out, const Kingdom & kingdom )
    {
        const PlayerColor ourColor = kingdom.GetColor();
        out << ",\"rivals\":[";
        bool first = true;
        for ( const PlayerColor color : PlayerColorsVector( Color::allPlayerColors() ) ) {
            if ( color == ourColor || ColorBase( color ).isFriends( ourColor ) || !world.GetKingdom( color ).isPlay() ) {
                continue;
            }
            for ( const Heroes * hero : world.GetKingdom( color ).GetHeroes() ) {
                const int32_t index = hero->GetIndex();
                if ( index < 0 || world.getTile( index ).isFog( ourColor ) ) {
                    continue;
                }
                const bool full = kingdom.Modes( Kingdom::IDENTIFYHERO ) || kingdom.IsTileVisibleFromCrystalBall( index );

                if ( !first ) {
                    out << ',';
                }
                first = false;
                out << "{\"c\":\"" << Color::String( color ) << "\",\"i\":" << index << ",\"full\":" << ( full ? 1 : 0 );
                const double estimate = writeArmy( out, hero->GetArmy(), full ? ArmyView::EXACT : ArmyView::BANDS );
                out << ",\"est\":" << static_cast<int64_t>( estimate );
                if ( full ) {
                    out << ",\"lvl\":" << hero->GetLevel() << ",\"a\":" << hero->GetAttack() << ",\"d\":" << hero->GetDefense() << ",\"pw\":" << hero->GetPower()
                        << ",\"k\":" << hero->GetKnowledge() << ",\"sp\":" << hero->GetSpellPoints() << ",\"mp\":" << hero->GetMovePoints()
                        << ",\"mor\":" << hero->GetMorale() << ",\"luck\":" << hero->GetLuck();
                }
                out << '}';
            }
        }
        out << ']';
    }
}

namespace
{
    // Everything a player sees about his own hero in the hero dialog.
    void writeOwnHero( std::ostringstream & out, const Heroes & hero )
    {
        out << "{\"id\":" << hero.GetID() << ",\"i\":" << hero.GetIndex() << ",\"mp\":" << hero.GetMovePoints() << ",\"mmp\":" << hero.GetMaxMovePoints()
            << ",\"str\":" << hero.GetArmy().GetStrength() << ",\"race\":" << hero.GetRace() << ",\"lvl\":" << hero.GetLevel() << ",\"a\":" << hero.GetAttack()
            << ",\"d\":" << hero.GetDefense() << ",\"pw\":" << hero.GetPower() << ",\"k\":" << hero.GetKnowledge() << ",\"sp\":" << hero.GetSpellPoints()
            << ",\"msp\":" << hero.GetMaxSpellPoints() << ",\"mor\":" << hero.GetMorale() << ",\"luck\":" << hero.GetLuck()
            << ",\"book\":" << ( hero.HaveSpellBook() ? 1 : 0 );
        writeArmy( out, hero.GetArmy(), ArmyView::EXACT );
        // Secondary skill levels, Skill::Secondary::PATHFINDING (1) .. ESTATES (14).
        out << ",\"sk\":[";
        for ( int skill = Skill::Secondary::PATHFINDING; skill <= Skill::Secondary::ESTATES; ++skill ) {
            out << ( skill > Skill::Secondary::PATHFINDING ? "," : "" ) << hero.GetLevelSkill( skill );
        }
        out << "],\"art\":[";
        bool first = true;
        for ( const Artifact & artifact : hero.GetBagArtifacts() ) {
            if ( !artifact.isValid() ) {
                continue;
            }
            out << ( first ? "" : "," ) << artifact.GetID();
            first = false;
        }
        out << "]}";
    }

    // Everything a player sees about his own castle: race, built buildings, the garrison and the
    // creatures available in each dwelling level (the monster of the best built dwelling).
    void writeOwnCastle( std::ostringstream & out, const Castle & castle )
    {
        out << "{\"n\":\"" << castle.GetName() << "\",\"i\":" << castle.GetIndex() << ",\"race\":" << castle.GetRace() << ",\"castle\":" << ( castle.isCastle() ? 1 : 0 )
            << ",\"b\":" << castle.getBuildingsMask();
        writeArmy( out, castle.GetArmy(), ArmyView::EXACT );
        out << ",\"dw\":[";
        const uint32_t dwellings[] = { DWELLING_MONSTER1, DWELLING_MONSTER2, DWELLING_MONSTER3, DWELLING_MONSTER4, DWELLING_MONSTER5, DWELLING_MONSTER6 };
        for ( size_t level = 0; level < std::size( dwellings ); ++level ) {
            const uint32_t count = castle.isBuild( dwellings[level] ) ? castle.getMonstersInDwelling( dwellings[level] ) : 0;
            out << ( level > 0 ? "," : "" );
            if ( castle.isBuild( dwellings[level] ) ) {
                writeMonster( out, Monster( castle.GetRace(), castle.GetActualDwelling( dwellings[level] ) ), count );
            }
            else {
                out << "[0,0,0,0,0,0,0]";
            }
        }
        out << "]}";
    }

    // Castles and towns of other owners (rivals and neutral) outside the fog, as the castle quick
    // info shows them (dialog_quickinfo.cpp): race, castle or town, the owner; the defenders are
    // unknown without a Thieves' Guild, monster types with one guild, size words with two or more,
    // exact counts in the Crystal Ball view.
    void writeVisibleCastles( std::ostringstream & out, const Kingdom & kingdom )
    {
        const PlayerColor ourColor = kingdom.GetColor();
        const uint32_t guilds = kingdom.GetCountThievesGuild();
        out << ",\"rcastles\":[";
        bool first = true;
        const int32_t size = world.w() * world.h();
        for ( int32_t index = 0; index < size; ++index ) {
            const Castle * castle = world.getCastleEntrance( Maps::GetPoint( index ) );
            if ( castle == nullptr || castle->GetIndex() != index || castle->isFriends( ourColor ) || world.getTile( index ).isFog( ourColor ) ) {
                continue;
            }
            ArmyView view = ArmyView::UNKNOWN;
            if ( kingdom.IsTileVisibleFromCrystalBall( index ) ) {
                view = ArmyView::EXACT;
            }
            else if ( guilds > 1 ) {
                view = ArmyView::BANDS;
            }
            else if ( guilds == 1 ) {
                view = ArmyView::TYPES;
            }
            out << ( first ? "" : "," ) << "{\"c\":\"" << Color::String( castle->GetColor() ) << "\",\"i\":" << index << ",\"race\":" << castle->GetRace()
                << ",\"castle\":" << ( castle->isCastle() ? 1 : 0 ) << ",\"vis\":" << static_cast<int>( view );
            first = false;
            const double estimate = writeArmy( out, castle->GetArmy(), view );
            out << ",\"est\":" << static_cast<int64_t>( estimate ) << '}';
        }
        out << ']';
    }
}

namespace
{
    // FHEROES2_RESEED="day:salt": before the first AI turn of `day` the game's random generator is
    // re-seeded and the world seed (battle seeds, obstacles, monster reactions) is shifted by the
    // salt. The game up to that moment stays byte-identical, everything after it gets other luck:
    // replays of one strategic answer under several salts measure how much of a label is chance
    // (rl/label_noise.py).
    void applyReseed()
    {
        static bool applied = false;
        static const std::pair<uint32_t, uint64_t> request = [] {
            const char * value = std::getenv( "FHEROES2_RESEED" );
            if ( value == nullptr || std::strchr( value, ':' ) == nullptr ) {
                return std::pair<uint32_t, uint64_t>( 0, 0 );
            }
            return std::pair<uint32_t, uint64_t>( static_cast<uint32_t>( std::strtoul( value, nullptr, 10 ) ),
                                                  std::strtoull( std::strchr( value, ':' ) + 1, nullptr, 10 ) );
        }();
        if ( applied || request.first == 0 || world.CountDay() < request.first ) {
            return;
        }
        applied = true;

        const uint64_t seed = ( request.second + 1 ) * 0x9E3779B97F4A7C15ULL ^ request.first;
        Rand::SeedCurrentThread( seed );
        world.SetMapSeed( world.GetMapSeed() ^ static_cast<uint32_t>( seed >> 32 ) );
    }
}

void AIDecision::sendTurnContext( const Kingdom & kingdom )
{
    if ( !isEnabled() ) {
        return;
    }

    applyReseed();
    sendDayReport();

    std::ostringstream out;
    // "p" uses the same color names as the "results" of the "game_end" event.
    out << "{\"ev\":\"turn_context\",\"t\":" << world.CountDay() << ",\"p\":\"" << Color::String( kingdom.GetColor() ) << "\",\"diff\":" << Game::getDifficulty();

    const Funds & funds = kingdom.GetFunds();
    out << ",\"res\":[" << funds.wood << ',' << funds.mercury << ',' << funds.ore << ',' << funds.sulfur << ',' << funds.crystal << ',' << funds.gems << ','
        << funds.gold << ']';

    // Day of the week (1-7; creatures grow on day 1) and the week.
    out << ",\"wd\":" << world.GetDay() << ",\"wk\":" << world.GetWeek();

    out << ",\"castles\":[";
    const VecCastles & castles = kingdom.GetCastles();
    for ( size_t i = 0; i < castles.size(); ++i ) {
        if ( i > 0 ) {
            out << ',';
        }
        writeOwnCastle( out, *castles[i] );
    }
    out << ']';

    out << ",\"heroes\":[";
    const VecHeroes & heroes = kingdom.GetHeroes();
    for ( size_t i = 0; i < heroes.size(); ++i ) {
        if ( i > 0 ) {
            out << ',';
        }
        writeOwnHero( out, *heroes[i] );
    }
    out << "]";

    writeVisibleRivals( out, kingdom );
    writeVisibleCastles( out, kingdom );
    out << ",\"w\":" << world.w() << "}";

    std::cout << out.str() << "\n";
    std::cout.flush();
}

int32_t AIDecision::requestHeroTarget( const Heroes & hero, const std::vector<AI::TargetCandidate> & candidates )
{
    if ( !isEnabled() ) {
        return -1;
    }

    std::ostringstream out;
    out << "{\"ev\":\"decision\",\"t\":" << world.CountDay() << ",\"p\":\"" << Color::String( hero.GetColor() ) << "\",\"h\":" << hero.GetID() << ",\"from\":" << hero.GetIndex() << ",\"cands\":[";
    for ( size_t i = 0; i < candidates.size(); ++i ) {
        if ( i > 0 ) {
            out << ',';
        }
        out << "{\"i\":" << candidates[i].index << ",\"obj\":" << static_cast<int>( candidates[i].objectType ) << ",\"v\":" << candidates[i].value
            << ",\"d\":" << candidates[i].distance << "}";
    }
    out << "]}";

    std::cout << out.str() << "\n";
    std::cout.flush();

    std::string line;
    while ( std::getline( std::cin, line ) ) {
        if ( line.find( "\"pick\"" ) != std::string::npos ) {
            const int64_t heroId = extractInt( line, "h", -1 );
            const int64_t tileIndex = extractInt( line, "i", -1 );

            if ( heroId != hero.GetID() ) {
                markChannelBroken();
                return -1;
            }

            for ( const AI::TargetCandidate & candidate : candidates ) {
                if ( candidate.index == tileIndex ) {
                    return static_cast<int32_t>( tileIndex );
                }
            }

            // The chosen tile is not a valid candidate: ignore this decision, keep the channel alive.
            return -1;
        }
        if ( line.find( "\"skip\"" ) != std::string::npos ) {
            return -1;
        }
        // Ignore unknown lines and keep waiting for a proper operation.
    }

    // The agent is gone.
    markChannelBroken();

    return -1;
}

int32_t AIDecision::requestBuild( const Castle & castle, const std::vector<BuildCandidate> & candidates, const bool defensive )
{
    if ( !isEnabled() || candidates.empty() ) {
        return replySkip;
    }

    const int race = castle.GetRace();

    std::ostringstream out;
    out << "{\"ev\":\"build\",\"t\":" << world.CountDay() << ",\"p\":\"" << Color::String( castle.GetColor() ) << "\",\"castle\":" << castle.GetIndex()
        << ",\"race\":" << race << ",\"defensive\":" << ( defensive ? 1 : 0 );
    writeFunds( out, "res", castle.GetKingdom().GetFunds() );
    out << ",\"cands\":[";
    for ( size_t i = 0; i < candidates.size(); ++i ) {
        if ( i > 0 ) {
            out << ',';
        }
        const uint32_t building = candidates[i].building;
        out << "{\"b\":" << building << ",\"name\":\"" << Castle::GetStringBuilding( building, race ) << "\",\"trade\":" << ( candidates[i].needsTrade ? 1 : 0 );
        writeFunds( out, "cost", PaymentConditions::BuyBuilding( race, building ) );
        out << '}';
    }
    out << "]}";

    std::cout << out.str() << "\n";
    std::cout.flush();

    std::string line;
    if ( !readReply( "\"build\"", line ) ) {
        return replySkip;
    }

    const int64_t building = extractInt( line, "b", -1 );
    if ( building == 0 ) {
        return replyNone;
    }
    for ( size_t i = 0; i < candidates.size(); ++i ) {
        if ( candidates[i].building == building ) {
            return static_cast<int32_t>( i );
        }
    }

    // Not a candidate (or a "skip"): the built-in AI decides, the channel stays alive.
    return replySkip;
}

void AIDecision::reportBuildResult( const Castle & castle, const uint32_t building, const bool byAgent )
{
    if ( !isEnabled() ) {
        return;
    }

    std::cout << "{\"ev\":\"build_result\",\"t\":" << world.CountDay() << ",\"p\":\"" << Color::String( castle.GetColor() ) << "\",\"castle\":"
              << castle.GetIndex() << ",\"b\":" << building << ",\"src\":\"" << ( byAgent ? "agent" : "builtin" ) << "\"}\n";
    std::cout.flush();
}

int32_t AIDecision::requestHire( const Kingdom & kingdom, const std::vector<HireCandidate> & candidates, const int32_t builtinChoice )
{
    if ( !isEnabled() || candidates.empty() ) {
        return replySkip;
    }

    std::ostringstream out;
    out << "{\"ev\":\"hire\",\"t\":" << world.CountDay() << ",\"p\":\"" << Color::String( kingdom.GetColor() ) << "\",\"heroes\":" << kingdom.GetHeroes().size();
    writeFunds( out, "res", kingdom.GetFunds() );
    out << ",\"cands\":[";
    for ( size_t i = 0; i < candidates.size(); ++i ) {
        if ( i > 0 ) {
            out << ',';
        }
        const HireCandidate & candidate = candidates[i];
        out << "{\"castle\":" << candidate.castle->GetIndex() << ",\"slot\":" << candidate.slot << ",\"hero\":" << candidate.hero->GetID()
            << ",\"race\":" << candidate.hero->GetRace() << ",\"lvl\":" << candidate.hero->GetLevel() << ",\"val\":" << candidate.hero->getRecruitValue()
            << ",\"army\":" << candidate.castle->getArmyRecruitmentValue() << '}';
    }
    out << "],\"bi\":" << builtinChoice << "}";

    std::cout << out.str() << "\n";
    std::cout.flush();

    std::string line;
    if ( !readReply( "\"hire\"", line ) ) {
        return replySkip;
    }

    const int64_t castleIndex = extractInt( line, "castle", -2 );
    if ( castleIndex == -1 ) {
        return replyNone;
    }
    const int64_t slot = extractInt( line, "slot", 0 );
    for ( size_t i = 0; i < candidates.size(); ++i ) {
        if ( candidates[i].castle->GetIndex() == castleIndex && candidates[i].slot == slot ) {
            return static_cast<int32_t>( i );
        }
    }

    return replySkip;
}

int32_t AIDecision::requestArmy( const Castle & castle, const char * reason, const std::vector<ArmyOffer> & offer )
{
    if ( !isEnabled() || offer.empty() ) {
        return replySkip;
    }

    const Heroes * guestHero = castle.GetHero();

    std::ostringstream out;
    out << "{\"ev\":\"army\",\"t\":" << world.CountDay() << ",\"p\":\"" << Color::String( castle.GetColor() ) << "\",\"castle\":" << castle.GetIndex()
        << ",\"reason\":\"" << reason << "\",\"guest\":" << ( guestHero ? guestHero->GetID() : -1 )
        << ",\"garrison\":" << castle.GetArmy().GetStrength() << ",\"hero\":" << ( guestHero ? guestHero->GetArmy().GetStrength() : 0.0 );
    writeFunds( out, "res", castle.GetKingdom().GetFunds() );
    out << ",\"offer\":[";
    for ( size_t i = 0; i < offer.size(); ++i ) {
        if ( i > 0 ) {
            out << ',';
        }
        out << "{\"mon\":" << offer[i].monsterId << ",\"avail\":" << offer[i].available << ",\"n\":" << offer[i].affordable << ",\"str\":" << offer[i].strength
            << '}';
    }
    out << "]}";

    std::cout << out.str() << "\n";
    std::cout.flush();

    std::string line;
    if ( !readReply( "\"army\"", line ) ) {
        return replySkip;
    }

    const int64_t percent = extractInt( line, "pct", -1 );
    if ( percent < 0 || percent > 100 ) {
        return replySkip;
    }

    return static_cast<int32_t>( percent );
}

void AIDecision::sendGameOver( const uint32_t playthroughId, const char * summaryJson )
{
    std::cout << "{\"ev\":\"game_end\",\"playthrough\":" << playthroughId << "," << summaryJson << "}\n";
    std::cout.flush();
}
