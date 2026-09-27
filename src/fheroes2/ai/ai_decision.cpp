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

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ai_planner.h"
#include "army.h"
#include "castle.h"
#include "color.h"
#include "game.h"
#include "heroes.h"
#include "kingdom.h"
#include "logging.h"
#include "maps_tiles.h"
#include "payment.h"
#include "resource.h"
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

void AIDecision::sendTurnContext( const Kingdom & kingdom )
{
    if ( !isEnabled() ) {
        return;
    }

    std::ostringstream out;
    // "p" uses the same color names as the "results" of the "game_end" event.
    out << "{\"ev\":\"turn_context\",\"t\":" << world.CountDay() << ",\"p\":\"" << Color::String( kingdom.GetColor() ) << "\",\"diff\":" << Game::getDifficulty();

    const Funds & funds = kingdom.GetFunds();
    out << ",\"res\":[" << funds.wood << ',' << funds.mercury << ',' << funds.ore << ',' << funds.sulfur << ',' << funds.crystal << ',' << funds.gems << ','
        << funds.gold << ']';

    out << ",\"castles\":[";
    const VecCastles & castles = kingdom.GetCastles();
    for ( size_t i = 0; i < castles.size(); ++i ) {
        if ( i > 0 ) {
            out << ',';
        }
        out << "{\"n\":\"" << castles[i]->GetName() << "\",\"i\":" << castles[i]->GetIndex() << "}";
    }
    out << ']';

    out << ",\"heroes\":[";
    const VecHeroes & heroes = kingdom.GetHeroes();
    for ( size_t i = 0; i < heroes.size(); ++i ) {
        if ( i > 0 ) {
            out << ',';
        }
        const Heroes * hero = heroes[i];
        out << "{\"id\":" << hero->GetID() << ",\"i\":" << hero->GetIndex() << ",\"mp\":" << hero->GetMovePoints() << ",\"mmp\":" << hero->GetMaxMovePoints()
            << ",\"str\":" << hero->GetArmy().GetStrength() << "}";
    }
    out << "]}";

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
