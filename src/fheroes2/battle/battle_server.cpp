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

#include "battle_server.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ai_battle.h"
#include "army.h"
#include "army_troop.h"
#include "battle.h"
#include "battle_arena.h"
#include "battle_army.h"
#include "battle_board.h"
#include "battle_bridge.h"
#include "battle_cell.h"
#include "battle_command.h"
#include "battle_tower.h"
#include "battle_troop.h"
#include "castle.h"
#include "color.h"
#include "game_auto_playtest.h"
#include "game_io.h"
#include "heroes.h"
#include "heroes_base.h"
#include "logging.h"
#include "maps.h"
#include "maps_fileinfo.h"
#include "maps_tiles.h"
#include "monster.h"
#include "mp2.h"
#include "players.h"
#include "rand.h"
#include "save_format_version.h"
#include "serialize.h"
#include "settings.h"
#include "spell.h"
#include "spell_storage.h"
#include "world.h"

namespace
{
    int64_t extractInt( const std::string & line, const char * key, const int64_t defaultValue )
    {
        const std::string needle = "\"" + std::string( key ) + "\":";
        const size_t pos = line.find( needle );
        if ( pos == std::string::npos ) {
            return defaultValue;
        }

        return std::strtoll( line.c_str() + pos + needle.size(), nullptr, 10 );
    }

    // Returns the offset of the first non-whitespace character after the "key": marker.
    size_t skipToValue( const std::string & line, const char * key )
    {
        const std::string needle = "\"" + std::string( key ) + "\":";
        const size_t pos = line.find( needle );
        if ( pos == std::string::npos ) {
            return std::string::npos;
        }

        size_t offset = pos + needle.size();
        while ( offset < line.size() && std::isspace( static_cast<unsigned char>( line[offset] ) ) ) {
            ++offset;
        }

        return offset;
    }

    std::string extractString( const std::string & line, const char * key )
    {
        const size_t start = skipToValue( line, key );
        if ( start == std::string::npos || start >= line.size() || line[start] != '"' ) {
            return {};
        }

        const size_t end = line.find( '"', start + 1 );
        if ( end == std::string::npos ) {
            return {};
        }

        return line.substr( start + 1, end - start - 1 );
    }

    std::vector<int64_t> extractIntArray( const std::string & line, const char * key )
    {
        std::vector<int64_t> result;

        const size_t start = skipToValue( line, key );
        if ( start == std::string::npos || start >= line.size() || line[start] != '[' ) {
            return result;
        }

        const size_t end = line.find( ']', start );
        if ( end == std::string::npos ) {
            return result;
        }

        const std::string body = line.substr( start, end - start );
        size_t offset = 0;
        while ( offset < body.size() ) {
            const char * beginPtr = body.c_str() + offset;
            char * endPtr = nullptr;
            const long long value = std::strtoll( beginPtr, &endPtr, 10 );
            if ( endPtr == beginPtr ) {
                ++offset;
                continue;
            }

            result.push_back( value );
            offset = static_cast<size_t>( endPtr - body.c_str() ) + 1;
        }

        return result;
    }

    // Unread bytes of a stream -> lowercase hex string.
    std::string encodeHex( const RWStreamBuf & stream )
    {
        static const char digits[] = "0123456789abcdef";
        const uint8_t * data = stream.data();
        const size_t size = stream.size();

        std::string result;
        result.reserve( size * 2 );
        for ( size_t i = 0; i < size; ++i ) {
            result.push_back( digits[data[i] >> 4] );
            result.push_back( digits[data[i] & 0x0F] );
        }

        return result;
    }

    // Hex string -> bytes; false on garbage (odd length, non-hex characters).
    bool decodeHex( const std::string & text, std::vector<uint8_t> & bytes )
    {
        bytes.clear();
        if ( text.size() % 2 != 0 ) {
            return false;
        }

        bytes.reserve( text.size() / 2 );
        for ( size_t i = 0; i < text.size(); i += 2 ) {
            uint8_t value = 0;
            const auto [ptr, errorCode] = std::from_chars( text.data() + i, text.data() + i + 2, value, 16 );
            if ( errorCode != std::errc() || ptr != text.data() + i + 2 ) {
                return false;
            }
            bytes.push_back( value );
        }

        return true;
    }

    // Commander of one side from the "new" operation: the real hero, restored from its
    // save-game serialization (see Battle::EncodeCommander()).
    struct CommanderSpec
    {
        int32_t heroId = -1;  // -1: no commander
        std::vector<uint8_t> data;
    };

    // "<side>hid" + "<side>hero" of the "new" operation; no/invalid data -> no commander.
    CommanderSpec parseCommander( const std::string & line, const char * idKey, const char * dataKey )
    {
        CommanderSpec spec;

        const int64_t heroId = extractInt( line, idKey, -1 );
        if ( heroId < 0 || !decodeHex( extractString( line, dataKey ), spec.data ) || spec.data.empty() ) {
            spec.data.clear();
            return spec;
        }

        spec.heroId = static_cast<int32_t>( heroId );
        return spec;
    }

    // Restores the hero of a CommanderSpec into the world's hero object with the same id (the
    // battle server loads the same map as the real game, so the object exists). Returns the
    // hero's army or nullptr on failure. Called at every battle rebuild: the whole hero state
    // (spell points, modes) goes back to the battle start.
    Army * restoreCommander( const CommanderSpec & spec )
    {
        if ( spec.heroId < 0 ) {
            return nullptr;
        }

        Heroes * hero = world.GetHeroes( spec.heroId );
        if ( hero == nullptr ) {
            return nullptr;
        }

        // The serialization is always produced by this very binary (current format).
        Game::SetVersionOfCurrentSaveFile( CURRENT_FORMAT_VERSION );

        ROStreamBuf stream( spec.data );
        stream >> *hero;
        if ( stream.fail() || hero->GetID() != spec.heroId ) {
            return nullptr;
        }

        return &hero->GetArmy();
    }

    // Restores the castle (or town) on the battle tile from its save-game serialization (see
    // Battle::EncodeCastle()): buildings (towers, moat, fortifications, captain's quarters), the
    // captain, the garrison and the owner. Returns nullptr on failure.
    Castle * restoreCastle( const std::vector<uint8_t> & data, const int32_t tileIndex )
    {
        if ( !Maps::isValidAbsIndex( tileIndex ) ) {
            return nullptr;
        }

        Castle * castle = world.getCastleEntrance( Maps::GetPoint( tileIndex ) );
        if ( castle == nullptr ) {
            return nullptr;
        }

        Game::SetVersionOfCurrentSaveFile( CURRENT_FORMAT_VERSION );

        ROStreamBuf stream( data );
        stream >> *castle;
        if ( stream.fail() || castle->GetIndex() != tileIndex ) {
            return nullptr;
        }

        return castle;
    }

    // Removes the heroes placed by the map from its tiles (once, after the map is loaded). A
    // tile with a hero reports the object under the hero through the HERO's state
    // (Tile::getMainObjectType()), so restoring a real-game hero that the map had put on a castle
    // entrance used to make that castle disappear for world.getCastleEntrance().
    void clearHeroesFromTiles()
    {
        const int32_t mapSize = world.getSize();
        for ( int32_t idx = 0; idx < mapSize; ++idx ) {
            Maps::Tile & tile = world.getTile( idx );
            if ( tile.getHero() != nullptr ) {
                tile.setHero( nullptr );
            }
        }
    }

    // Takes every hero of the world off the map. The battle server's world is a replica of the
    // real game only in the heroes it restores for a battle: a hero left on a castle entrance by
    // the map (or by an earlier battle) would otherwise be taken for the castle's hero
    // (Castle::GetHero() and Heroes::inCastle() look heroes up by position).
    void parkAllHeroes()
    {
        for ( int heroId = 0; heroId < 256; ++heroId ) {
            if ( Heroes * hero = world.GetHeroes( heroId ); hero != nullptr ) {
                hero->SetCenter( { -1, -1 } );
            }
        }
    }

    // World seed of the battle-server mode (see RunBattleServer()).
    const uint32_t pinnedWorldSeed = 20260926;

    // One army stack from the "new" operation. `slot` is the army slot index the stack must
    // occupy (positions on the battle board derive from it); -1 means "first free slot" (the
    // plain "mon x count" format). Real-battle replication (see battle_agent.cpp) needs
    // explicit slots because real armies may have gaps.
    struct StackSpec
    {
        int32_t slot = -1;
        int32_t mon = 0;
        uint32_t count = 0;
    };

    // Parses the whole `text` as a decimal integer; false on garbage (no exceptions: a malformed
    // request must not bring the engine down).
    bool parseWholeInt( const std::string_view text, int32_t & value )
    {
        const char * end = text.data() + text.size();
        const auto [ptr, errorCode] = std::from_chars( text.data(), end, value );
        return errorCode == std::errc() && ptr == end && !text.empty();
    }

    // "13x10,21x24" or with explicit army slots "0:13x10,2:21x24" -> StackSpec list. Malformed
    // tokens are skipped.
    std::vector<StackSpec> parseStacks( const std::string & text )
    {
        std::vector<StackSpec> result;

        size_t offset = 0;
        while ( offset < text.size() ) {
            const size_t next = text.find( ',', offset );
            const std::string_view token = std::string_view{ text }.substr( offset, ( next == std::string::npos ? text.size() : next ) - offset );

            StackSpec spec;
            bool valid = true;

            std::string_view rest = token;
            const size_t slotSep = token.find( ':' );
            if ( slotSep != std::string_view::npos ) {
                valid = parseWholeInt( token.substr( 0, slotSep ), spec.slot ) && spec.slot >= 0;
                rest = token.substr( slotSep + 1 );
            }

            const size_t sep = rest.find( 'x' );
            int32_t count = 0;
            if ( valid && sep != std::string_view::npos && parseWholeInt( rest.substr( 0, sep ), spec.mon ) && parseWholeInt( rest.substr( sep + 1 ), count )
                 && count > 0 ) {
                spec.count = static_cast<uint32_t>( count );
                result.push_back( spec );
            }

            if ( next == std::string::npos ) {
                break;
            }
            offset = next + 1;
        }

        return result;
    }
}

namespace Battle
{
    // Thrown by the action provider when the replay command queue is exhausted; unwinds the
    // battle and reports the current state to the client.
    struct PauseBattle
    {};

    namespace
    {
        // A client command is accepted only for the unit to move and only if the engine would apply
        // it (ApplyAction*() silently drops invalid commands in Release builds and asserts in Debug).
        // Only the command types the legal-move enumeration produces are accepted.
        bool isAcceptableCommand( const Battle::Unit & unit, const Battle::Command & cmd )
        {
            const auto uid = static_cast<int>( unit.GetUID() );

            // Decode exactly like ApplyAction*() does: GetNextValue() on a copy (the values are
            // stored in reverse constructor order).
            Battle::Command values = cmd;

            switch ( cmd.GetType() ) {
            case Battle::CommandType::MOVE: {
                if ( cmd.size() != 2 || values.GetNextValue() != uid ) {
                    return false;
                }
                const int32_t dst = values.GetNextValue();
                return Battle::Arena::isValidMoveCommand( unit, dst );
            }
            case Battle::CommandType::ATTACK: {
                if ( cmd.size() != 5 || values.GetNextValue() != uid ) {
                    return false;
                }
                const Battle::Unit * defender = Battle::GetArena()->GetTroopUID( static_cast<uint32_t>( values.GetNextValue() ) );
                const int32_t dst = values.GetNextValue();
                const int32_t tgt = values.GetNextValue();
                const int dir = values.GetNextValue();
                return defender != nullptr && Battle::Arena::isValidAttackCommand( unit, *defender, dst, tgt, dir );
            }
            case Battle::CommandType::SKIP:
                return cmd.size() == 1 && values.GetNextValue() == uid;
            case Battle::CommandType::SPELLCAST: {
                // A hero spell is legal exactly when the enumeration offers it (the targeting rules
                // live there, mirroring the battle interface).
                const std::vector<Command> casts = EnumerateSpellCasts( *Battle::GetArena() );
                return std::any_of( casts.begin(), casts.end(), [&cmd]( const Command & cast ) {
                    return cast.size() == cmd.size() && std::equal( cast.begin(), cast.end(), cmd.begin() );
                } );
            }
            default:
                return false;
            }
        }

        // Parses a batched action path ("acts"/"lens"/"args" arrays, as sent by the "replay" and
        // "restore" operations) into engine commands.
        std::vector<Command> parseCommandPath( const std::string & line )
        {
            const std::vector<int64_t> acts = extractIntArray( line, "acts" );
            const std::vector<int64_t> lens = extractIntArray( line, "lens" );
            const std::vector<int64_t> args = extractIntArray( line, "args" );

            std::vector<Command> queue;
            queue.reserve( acts.size() );

            size_t argPos = 0;
            for ( size_t i = 0; i < acts.size(); ++i ) {
                const size_t count = ( i < lens.size() ) ? static_cast<size_t>( lens[i] ) : 0;

                std::vector<int> rawValues;
                rawValues.reserve( count );
                for ( size_t j = 0; j < count && argPos < args.size(); ++j, ++argPos ) {
                    rawValues.push_back( static_cast<int>( args[argPos] ) );
                }

                queue.push_back( Command::FromRaw( static_cast<CommandType>( acts[i] ), rawValues ) );
            }

            return queue;
        }

        // {"act":..,"args":[..]} of a command (wire order: the values as stored).
        void writeCommand( std::ostream & out, const Command & cmd )
        {
            out << "{\"act\":" << static_cast<int>( cmd.GetType() ) << ",\"args\":[";
            for ( size_t i = 0; i < cmd.size(); ++i ) {
                if ( i > 0 ) {
                    out << ',';
                }
                out << cmd[i];
            }
            out << "]}";
        }

        // The first command the built-in battle AI chooses for the unit (asked exactly like
        // Arena::UnitTurn() asks it for AI-controlled units), expressed as the equal enumerated legal
        // move when there is one: the AI leaves the target cell/direction of attacks for the engine
        // to resolve (-1), which the legal list spells out. Empty when the AI has no action.
        std::optional<Command> builtinChoice( Arena & arena, const Unit & unit, const std::vector<Command> & legalMoves )
        {
            Actions chosen;
            AI::BattlePlanner::Get().BattleTurn( arena, unit, chosen );
            if ( chosen.empty() ) {
                return std::nullopt;
            }

            const Command & expert = chosen.front();
            const auto sameCommand = []( const Command & lhs, const Command & rhs ) {
                return lhs.GetType() == rhs.GetType() && lhs.size() == rhs.size() && std::equal( lhs.begin(), lhs.end(), rhs.begin() );
            };

            for ( const Command & legal : legalMoves ) {
                if ( sameCommand( legal, expert ) ) {
                    return legal;
                }
            }

            if ( expert.GetType() == CommandType::ATTACK ) {
                const Command resolved = Arena::resolveAttackCommand( expert );
                for ( const Command & legal : legalMoves ) {
                    if ( legal.GetType() == CommandType::ATTACK && sameCommand( Arena::resolveAttackCommand( legal ), resolved ) ) {
                        return legal;
                    }
                }
            }

            return expert;
        }
    }

    class BattleServer
    {
    public:
        // Commanders (optional) replace the stacks of their side: the battle army is the hero's
        // own army. Colors < 0 keep the defaults (attacker RED, defender BLUE); a commander's
        // army always has the hero's color.
        bool newBattle( const uint32_t seed, const std::vector<StackSpec> & attackingStacks, const std::vector<StackSpec> & defendingStacks, int32_t tileIndex,
                        const bool attackingSpreadFormation, const bool defendingSpreadFormation, const CommanderSpec & attackingCommander = {},
                        const CommanderSpec & defendingCommander = {}, const int attackingColor = -1, const int defendingColor = -1,
                        const std::vector<uint8_t> & castleData = {}, const bool defendingGarrison = false, const int attackingScale = 100,
                        const int defendingScale = 100 );

        // Rebuilds the arena at the battle root; false when a commander cannot be restored.
        bool resetBattle();

        // Plays the current battle to the end, exchanging actions with the client at every unit
        // activation (see the protocol in rl/README.md).
        void play();

        // Applies the given action sequence from the battle root inside the engine (one
        // roundtrip for the whole path). With extendPath the sequence is appended to the main
        // line (the "action" operation); otherwise the main line is untouched (MCTS replays).
        // forceRebuild: skip the main-line-end snapshot and replay from the battle root (the reference path, used by tests).
        void replay( const std::vector<Command> & actionQueue, const bool extendPath, const bool forceRebuild = false );

        // Resets the main line to the battle root and reports the root state.
        void resetLine();

        // Plays the current battle to the end with the built-in battle AI, streaming one expert
        // record per decision: the full state (with legal moves) plus the action chosen by the
        // built-in AI. Used to generate training data from the ready-made algorithms.
        void runAuto();

        // Applies the given actions from the current battle root, pausing at the next decision
        // point (or when the battle ends). With resumeCurrentRound the battle is resumed where
        // a snapshot restore left it (mid-round) instead of starting a fresh turn.
        // Commands before `validateFrom` were already validated when they entered the main line; re-validating the whole
        // path on every replay made long battles quadratic in pathfinder work.
        void advance( const std::vector<Command> & path, const bool resumeCurrentRound = false, const size_t validateFrom = 0 );

        // Snapshot/restore of the battle state for the search tree: "snap" stores the current
        // pause-point state under a client-chosen id, "restore" rewinds to it (optionally
        // applying an action path suffix and saving the result under another id) in one
        // roundtrip. Snapshots live inside the Arena and die with it ("new"/"reset").
        void snapshotSave( const int32_t id );
        // With `rollout` the battle then continues with the built-in AI on both sides until it ends
        // (or a round cap) and the final state is reported: counterfactual evaluation of a move
        // in one roundtrip (rl/battle_prefs.py). The arena is left at that final state; main-line
        // operations restore the main line themselves, other clients restore a snapshot first.
        void snapshotRestore( const int32_t id, const int32_t saveAsId, const std::vector<Command> & path, const bool rollout = false );
        void snapshotsClear();

        // Reports the current state with the action the built-in battle AI would take for the
        // unit to move (in the "expert" field); the action is not applied. Lets an external
        // agent play against the built-in AI in the gate runner (rl/gate.py).
        void suggest();

        // Reports the current state: a decision point (with legal moves) or the final result.
        void emitState();

        // The reply to an operation whose action path contained an illegal command.
        static void emitIllegalAction()
        {
            std::cout << "{\"ev\":\"error\",\"what\":\"illegal action\"}\n";
            std::cout.flush();
        }

        bool isQuitRequested() const
        {
            return _quitRequested;
        }

    private:
        Army _attackingArmy;
        Army _defendingArmy;
        std::unique_ptr<Rand::PCG32> _randomGenerator;
        std::unique_ptr<Arena> _arena;

        // Battle-state snapshots keyed by the client-chosen id. They hold plain data only, so
        // they stay valid across the arena rebuilds that every main-line operation performs;
        // a "new" battle (different setup) invalidates them.
        std::map<int32_t, std::shared_ptr<ArenaSnapshot>> _snapshots;

        // The state at the end of the main line (its current pause point). Main-line operations restore it and apply only the
        // new commands instead of replaying the whole main line from the battle root — replaying made long battles quadratic
        // (the real-battle agent mirrors every move and every MCTS search starts with a main-line replay). Null when the battle
        // is over; then the replay-from-root path is used. Restore + suffix == full replay (tested).
        std::shared_ptr<ArenaSnapshot> _mainLineEnd;

        void captureMainLineEnd()
        {
            _mainLineEnd = _arena->BattleValid() ? _arena->captureSnapshot() : nullptr;
        }

        bool restoreMainLineEnd()
        {
            return _mainLineEnd != nullptr && _arena->applySnapshot( *_mainLineEnd );
        }

        // Setup of the current battle, used by resetBattle() for replay-based search.
        uint32_t _seed = 0;
        int32_t _tileIndex = -1;
        std::vector<StackSpec> _attackingStacks;
        std::vector<StackSpec> _defendingStacks;
        bool _attackingSpreadFormation = true;
        bool _defendingSpreadFormation = true;
        CommanderSpec _attackingCommander;
        CommanderSpec _defendingCommander;
        int _attackingColor = -1;
        int _defendingColor = -1;
        // The castle/town on the battle tile (empty: keep the map's castle as loaded) and whether
        // the defenders are its garrison.
        std::vector<uint8_t> _castleData;
        bool _defendingGarrison = false;
        // Every stack of a side at this percentage of its count (rounded, at least 1 creature): the
        // handicap of the strategic duel label (rl/strategy_games.py final_duel_label).
        int _attackingScale = 100;
        int _defendingScale = 100;

        bool _quitRequested = false;

        // Set by advance() when the path contained a command the engine would reject.
        bool _illegalAction = false;

        // Actions applied since the battle root; the main line is replayed from scratch on
        // every "action" operation (cheap for the engine, keeps the protocol stateless).
        std::vector<Command> _currentPath;
    };

    bool BattleServer::newBattle( const uint32_t seed, const std::vector<StackSpec> & attackingStacks, const std::vector<StackSpec> & defendingStacks,
                                  int32_t tileIndex, const bool attackingSpreadFormation, const bool defendingSpreadFormation, const CommanderSpec & attackingCommander,
                                  const CommanderSpec & defendingCommander, const int attackingColor, const int defendingColor,
                                  const std::vector<uint8_t> & castleData, const bool defendingGarrison, const int attackingScale,
                                  const int defendingScale )
    {
        _seed = seed;
        _attackingStacks = attackingStacks;
        _defendingStacks = defendingStacks;
        _attackingSpreadFormation = attackingSpreadFormation;
        _defendingSpreadFormation = defendingSpreadFormation;
        _attackingCommander = attackingCommander;
        _defendingCommander = defendingCommander;
        _attackingColor = attackingColor;
        _defendingColor = defendingColor;
        _castleData = castleData;
        _defendingGarrison = defendingGarrison;
        _attackingScale = attackingScale;
        _defendingScale = defendingScale;
        _quitRequested = false;
        _snapshots.clear();  // the battle setup changed: stored snapshots are invalid

        if ( tileIndex < 0 ) {
            // Deterministically pick an open land tile without a castle.
            std::vector<int32_t> candidates;
            const int32_t mapSize = world.getSize();
            for ( int32_t idx = 0; idx < mapSize; ++idx ) {
                const Maps::Tile & tile = world.getTile( idx );
                if ( tile.isWater() || world.getCastleEntrance( Maps::GetPoint( idx ) ) != nullptr ) {
                    continue;
                }

                candidates.push_back( idx );
            }

            if ( candidates.empty() ) {
                return false;
            }

            tileIndex = candidates[_seed % candidates.size()];
        }

        _tileIndex = tileIndex;
        _currentPath.clear();

        // Only the heroes of this battle may stand on the map (see parkAllHeroes()).
        parkAllHeroes();

        if ( !resetBattle() ) {
            // Leave a valid (commander-less) arena behind: every other operation needs one.
            _attackingCommander = {};
            _defendingCommander = {};
            _castleData.clear();
            _defendingGarrison = false;
            resetBattle();
            _mainLineEnd = nullptr;
            return false;
        }
        advance( {} );
        captureMainLineEnd();
        emitState();

        return true;
    }

    bool BattleServer::resetBattle()
    {
        _attackingArmy.Reset();
        _defendingArmy.Reset();

        // Destroy the old arena first: only one Arena instance may exist at a time (the class
        // keeps a static pointer to the current instance). The commanders are restored below, and
        // the old arena must not outlive the hero state it refers to.
        _arena.reset();

        // The castle first: the heroes' castle modifiers look it up.
        Castle * castle = nullptr;
        if ( !_castleData.empty() ) {
            castle = restoreCastle( _castleData, _tileIndex );
            if ( castle == nullptr ) {
                return false;
            }
        }

        Army * attackingArmy = &_attackingArmy;
        Army * defendingArmy = &_defendingArmy;
        if ( _defendingGarrison ) {
            if ( castle == nullptr ) {
                return false;
            }
            defendingArmy = &castle->GetArmy();
        }
        if ( _attackingCommander.heroId >= 0 ) {
            attackingArmy = restoreCommander( _attackingCommander );
        }
        if ( _defendingCommander.heroId >= 0 ) {
            defendingArmy = restoreCommander( _defendingCommander );
        }
        if ( attackingArmy == nullptr || defendingArmy == nullptr || attackingArmy == defendingArmy ) {
            return false;
        }

        for ( const StackSpec & spec : _attackingStacks ) {
            if ( spec.slot >= 0 && static_cast<size_t>( spec.slot ) < _attackingArmy.Size() ) {
                _attackingArmy.GetTroop( static_cast<size_t>( spec.slot ) )->Set( Monster( spec.mon ), spec.count );
            }
            else {
                _attackingArmy.AssignToFirstFreeSlot( Troop( Monster( spec.mon ), spec.count ), spec.count );
            }
        }

        for ( const StackSpec & spec : _defendingStacks ) {
            if ( spec.slot >= 0 && static_cast<size_t>( spec.slot ) < _defendingArmy.Size() ) {
                _defendingArmy.GetTroop( static_cast<size_t>( spec.slot ) )->Set( Monster( spec.mon ), spec.count );
            }
            else {
                _defendingArmy.AssignToFirstFreeSlot( Troop( Monster( spec.mon ), spec.count ), spec.count );
            }
        }

        // The handicap (scaled counts). Every rebuild starts from the restored commander / castle / stacks,
        // so the scaling never accumulates.
        for ( const auto & [army, scale] : { std::pair<Army *, int>{ attackingArmy, _attackingScale }, std::pair<Army *, int>{ defendingArmy, _defendingScale } } ) {
            if ( scale == 100 ) {
                continue;
            }
            for ( size_t index = 0; index < army->Size(); ++index ) {
                Troop * troop = army->GetTroop( index );
                if ( troop != nullptr && troop->isValid() ) {
                    const uint64_t scaled = ( static_cast<uint64_t>( troop->GetCount() ) * static_cast<uint64_t>( scale ) + 50 ) / 100;
                    troop->SetCount( static_cast<uint32_t>( std::max<uint64_t>( scaled, 1 ) ) );
                }
            }
        }

        // A hero's (or a garrison's) army carries its color and formation from the serialization.
        if ( attackingArmy == &_attackingArmy ) {
            _attackingArmy.SetColor( _attackingColor >= 0 ? static_cast<PlayerColor>( _attackingColor ) : PlayerColor::RED );
            _attackingArmy.SetSpreadFormation( _attackingSpreadFormation );
        }
        if ( defendingArmy == &_defendingArmy ) {
            _defendingArmy.SetColor( _defendingColor >= 0 ? static_cast<PlayerColor>( _defendingColor ) : PlayerColor::BLUE );
            _defendingArmy.SetSpreadFormation( _defendingSpreadFormation );
        }

        _randomGenerator = std::make_unique<Rand::PCG32>( _seed );
        _arena = std::make_unique<Arena>( *attackingArmy, *defendingArmy, _tileIndex, false, *_randomGenerator );

        return true;
    }

    void BattleServer::resetLine()
    {
        _currentPath.clear();
        resetBattle();
        advance( {} );
        captureMainLineEnd();
        emitState();
    }

    void BattleServer::advance( const std::vector<Command> & path, const bool resumeCurrentRound, const size_t validateFrom )
    {
        std::vector<Command> queue = path;

        // The provider feeds the queued actions to the engine; when the queue is exhausted the
        // battle is paused at the next decision point (PauseBattle unwinds the simulation).
        _illegalAction = false;

        auto provider = [this, &queue, &path, validateFrom]( Actions & actions ) {
            if ( queue.empty() ) {
                throw PauseBattle{};
            }

            const Unit * unit = _arena->getCurrentUnit();
            const bool isNewCommand = ( path.size() - queue.size() >= validateFrom );
            if ( unit == nullptr || ( isNewCommand && !isAcceptableCommand( *unit, queue.front() ) ) ) {
                // Stop at the decision point the illegal command was meant for; the caller
                // reports an error instead of a state.
                _illegalAction = true;
                throw PauseBattle{};
            }

            actions.push_back( queue.front() );
            queue.erase( queue.begin() );

            return true;
        };

        try {
            bool resume = resumeCurrentRound;
            while ( _arena->BattleValid() ) {
                if ( resume ) {
                    // Continue the interrupted round after a snapshot restore; the following
                    // rounds (if the path spans them) are regular Turns() calls.
                    _arena->resumeRound( provider );
                    resume = false;
                }
                else {
                    _arena->Turns( provider );
                }
            }
        }
        catch ( const PauseBattle & ) {
            // Pause: the state reply carries the legal moves for the unit to move.
        }
    }

    void BattleServer::snapshotSave( const int32_t id )
    {
        _snapshots[id] = _arena->captureSnapshot();
        emitState();
    }

    void BattleServer::snapshotRestore( const int32_t id, const int32_t saveAsId, const std::vector<Command> & path, const bool rollout )
    {
        const auto snapshotIter = _snapshots.find( id );
        if ( snapshotIter == _snapshots.end() || !_arena->applySnapshot( *snapshotIter->second ) ) {
            std::cout << "{\"ev\":\"error\",\"what\":\"unknown snapshot id\"}\n";
            std::cout.flush();
            return;
        }

        advance( path, true );

        if ( _illegalAction ) {
            emitIllegalAction();
            return;
        }

        if ( saveAsId > 0 ) {
            _snapshots[saveAsId] = _arena->captureSnapshot();
        }

        if ( rollout && _arena->BattleValid() ) {
            // Every pause point is mid-round: resume it the way a snapshot restore does (re-applying
            // the pause-point state keeps this on the tested restore path), then play on.
            const std::shared_ptr<ArenaSnapshot> pausePoint = _arena->captureSnapshot();
            _arena->applySnapshot( *pausePoint );

            auto provider = [this]( Actions & actions ) {
                const Unit * unit = _arena->getCurrentUnit();
                if ( unit == nullptr ) {
                    return false;
                }
                const std::optional<Command> expert = builtinChoice( *_arena, *unit, EnumerateLegalMoves( *_arena, *unit ) );
                if ( !expert ) {
                    return false;
                }
                actions.push_back( *expert );
                return true;
            };

            // The same cap as runAuto(): some matchups make the built-in planner loop forever.
            constexpr int32_t maxRolloutRounds = 200;
            _arena->resumeRound( provider );
            for ( int32_t rounds = 0; _arena->BattleValid() && rounds < maxRolloutRounds; ++rounds ) {
                _arena->Turns( provider );
            }
        }

        emitState();
    }

    void BattleServer::snapshotsClear()
    {
        _snapshots.clear();
        emitState();
    }

    void BattleServer::suggest()
    {
        if ( !_arena->BattleValid() ) {
            emitState();
            return;
        }

        const Unit * unit = _arena->getCurrentUnit();
        const std::vector<Command> legalMoves = ( unit != nullptr ? EnumerateLegalMoves( *_arena, *unit ) : std::vector<Command>{} );
        std::cout << SerializeArenaState( *_arena, unit, legalMoves );

        if ( unit != nullptr ) {
            // The built-in AI's action (like runAuto() records it), not applied: the client decides.
            if ( const std::optional<Command> expert = builtinChoice( *_arena, *unit, legalMoves ); expert ) {
                std::cout << ",\"expert\":";
                writeCommand( std::cout, *expert );
            }
        }

        std::cout << "}\n";
        std::cout.flush();
    }

    void BattleServer::runAuto()
    {
        auto provider = [this]( Actions & actions ) {
            const Unit * unit = _arena->getCurrentUnit();
            if ( unit == nullptr ) {
                return false;
            }

            const std::vector<Command> legalMoves = EnumerateLegalMoves( *_arena, *unit );

            const std::optional<Command> expert = builtinChoice( *_arena, *unit, legalMoves );
            if ( !expert ) {
                return false;
            }

            // Expert record: the pre-decision state (with legal moves) + the built-in AI's action.
            std::cout << SerializeArenaState( *_arena, unit, legalMoves ) << ",\"expert\":";
            writeCommand( std::cout, *expert );
            std::cout << "}\n";
            std::cout.flush();

            // One command per decision, like the agent protocol: after a spell the same unit
            // decides again (the built-in AI plans the unit's action anew).
            actions.push_back( *expert );

            return true;
        };

        // Safety cap: some army matchups make the built-in planner loop forever; in real games
        // this is prevented by the turn limit logic, so cap the rounds here explicitly.
        constexpr int32_t maxAutoRounds = 200;

        int32_t rounds = 0;
        while ( _arena->BattleValid() && rounds < maxAutoRounds ) {
            _arena->Turns( provider );
            ++rounds;
        }

        emitState();
    }

    void BattleServer::emitState()
    {
        if ( _arena->BattleValid() ) {
            const Unit * unit = _arena->getCurrentUnit();
            std::cout << SerializeArenaState( *_arena, unit, ( unit != nullptr ? EnumerateLegalMoves( *_arena, *unit ) : std::vector<Command>{} ) );
        }
        else {
            std::cout << SerializeArenaState( *_arena, nullptr, {} );
        }

        std::cout << "}\n";
        std::cout.flush();
    }

    void BattleServer::replay( const std::vector<Command> & actionQueue, const bool extendPath, const bool forceRebuild )
    {
        // The battle is deterministic, so any state materializes by rebuilding the arena at the
        // battle root and applying the whole action path inside the engine (one roundtrip); when the
        // main-line end snapshot is available, restoring it + applying the queue is equivalent and O(queue).
        // - extendPath (the "action" operation): the queue extends the main line, which is then
        //   replayed and becomes the new main line.
        // - otherwise (the batched search "replay"): the main line is untouched; the queue is
        //   applied on top of it, materializing the state the search asked for.
        if ( !forceRebuild && restoreMainLineEnd() ) {
            // Fast path: continue from the main-line end; every command of the queue is new and gets validated.
            advance( actionQueue, true );

            if ( extendPath && !_illegalAction ) {
                _currentPath.insert( _currentPath.end(), actionQueue.begin(), actionQueue.end() );
                captureMainLineEnd();
            }
        }
        else if ( extendPath ) {
            const size_t mainLineLength = _currentPath.size();
            _currentPath.insert( _currentPath.end(), actionQueue.begin(), actionQueue.end() );

            resetBattle();
            advance( _currentPath, false, mainLineLength );

            if ( _illegalAction ) {
                // The main line stays as it was (the arena is paused at its end).
                _currentPath.erase( _currentPath.begin() + static_cast<std::ptrdiff_t>( mainLineLength ), _currentPath.end() );
            }
            else {
                captureMainLineEnd();
            }
        }
        else {
            std::vector<Command> path = _currentPath;
            path.insert( path.end(), actionQueue.begin(), actionQueue.end() );

            resetBattle();
            advance( path, false, _currentPath.size() );
        }

        if ( _illegalAction ) {
            emitIllegalAction();
            return;
        }

        emitState();
    }

    std::string SerializeArenaState( Arena & arena, const Unit * currentUnit, const std::vector<Command> & legalMoves )
    {
        std::ostringstream out;
        out << "{\"ev\":\"state\",\"turn\":" << arena.GetTurnNumber()
            << ",\"cur\":" << ( currentUnit != nullptr ? static_cast<int64_t>( currentUnit->GetUID() ) : -1 );

        out << ",\"units\":[";
        bool firstUnit = true;
        for ( const int side : { 0, 1 } ) {
            const Force & force = ( side == 0 ) ? arena.getAttackingForce() : arena.getDefendingForce();
            const char * sideName = ( side == 0 ) ? "att" : "def";

            for ( const Unit * unit : force ) {
                if ( unit == nullptr || !unit->isValid() ) {
                    continue;
                }

                if ( !firstUnit ) {
                    out << ',';
                }
                firstUnit = false;

                out << "{\"u\":" << unit->GetUID() << ",\"side\":\"" << sideName << "\",\"mon\":" << unit->GetID() << ",\"q\":" << unit->GetCount()
                    << ",\"hpl\":" << unit->GetHitPointsLeft() << ",\"i\":" << unit->GetHeadIndex() << ",\"ti\":" << ( unit->isWide() ? unit->GetTailIndex() : -1 )
                    << ",\"sp\":" << unit->GetSpeed( true, false ) << ",\"shots\":" << unit->GetShots() << ",\"moved\":" << ( unit->Modes( TR_MOVED ) ? 1 : 0 )
                    // Strength of the stack (monster strength x count, the measure of army strength in
                    // game_end): values the survivors of counterfactual rollouts (rl/battle_prefs.py).
                    << ",\"str\":" << static_cast<int64_t>( unit->Troop::GetStrength() + 0.5 )
                    << "}";
            }
        }
        out << ']';

        out << ",\"obstacles\":[";
        {
            const Board * board = Arena::GetBoard();
            bool firstObstacle = true;
            if ( board != nullptr ) {
                for ( const Cell & cell : *board ) {
                    if ( cell.GetObject() != 0 ) {
                        if ( !firstObstacle ) {
                            out << ',';
                        }
                        firstObstacle = false;

                        out << cell.GetIndex();
                    }
                }
            }
        }
        out << ']';

        // Commanders: the spell points and whether the side already cast a spell this round (the
        // only commander state a battle changes). Present only for sides with a commander.
        out << ",\"heroes\":[";
        {
            bool firstHero = true;
            for ( const int side : { 0, 1 } ) {
                const Force & force = ( side == 0 ) ? arena.getAttackingForce() : arena.getDefendingForce();
                const HeroBase * commander = force.GetCommander();
                if ( commander == nullptr ) {
                    continue;
                }
                if ( !firstHero ) {
                    out << ',';
                }
                firstHero = false;
                out << "{\"side\":\"" << ( side == 0 ? "att" : "def" ) << "\",\"sp\":" << commander->GetSpellPoints()
                    << ",\"cast\":" << ( commander->Modes( Heroes::SPELLCASTED ) ? 1 : 0 ) << '}';
            }
        }
        out << ']';

        // Sieges: wall/tower cell states (board objects), towers (1 standing, 0 destroyed, -1 not
        // built) and the bridge (0 up, 1 down, 2 destroyed).
        if ( Arena::GetCastle() != nullptr ) {
            out << ",\"siege\":{\"cells\":[";
            bool firstCell = true;
            for ( const Cell & cell : *Arena::GetBoard() ) {
                if ( cell.GetObject() != 0 ) {
                    if ( !firstCell ) {
                        out << ',';
                    }
                    firstCell = false;
                    out << '[' << cell.GetIndex() << ',' << cell.GetObject() << ']';
                }
            }
            out << "],\"towers\":[";
            bool firstTower = true;
            for ( const TowerType type : { TowerType::TWR_LEFT, TowerType::TWR_CENTER, TowerType::TWR_RIGHT } ) {
                const Tower * tower = Arena::GetTower( type );
                if ( !firstTower ) {
                    out << ',';
                }
                firstTower = false;
                out << ( tower == nullptr ? -1 : ( tower->isValid() ? 1 : 0 ) );
            }
            const Bridge * bridge = Arena::GetBridge();
            out << "],\"bridge\":" << ( bridge == nullptr ? -1 : ( bridge->isDestroyed() ? 2 : ( bridge->isDown() ? 1 : 0 ) ) ) << '}';
        }

        if ( currentUnit != nullptr ) {
            out << ",\"legal\":[";
            for ( size_t i = 0; i < legalMoves.size(); ++i ) {
                const Command & cmd = legalMoves[i];
                if ( i > 0 ) {
                    out << ',';
                }

                out << "{\"act\":" << static_cast<int>( cmd.GetType() ) << ",\"args\":[";
                for ( size_t j = 0; j < cmd.size(); ++j ) {
                    if ( j > 0 ) {
                        out << ',';
                    }
                    out << cmd[j];
                }
                out << "]}";
            }
            out << ']';
        }

        if ( !arena.BattleValid() ) {
            const Result & result = arena.GetResult();
            const char * winner = "draw";
            if ( result.attacker & RESULT_WINS ) {
                winner = "att";
            }
            else if ( result.defender & RESULT_WINS ) {
                winner = "def";
            }

            out << ",\"result\":\"" << winner << "\"";
        }

        return out.str();
    }

    std::vector<Command> EnumerateLegalMoves( Arena & arena, const Unit & unit )
    {
        std::vector<Command> moves;

        const uint32_t uid = unit.GetUID();

        // All cells reachable by the unit's head on the current turn, plus the current position.
        std::vector<int32_t> cells = arena.getAllAvailableMoves( unit );
        cells.push_back( unit.GetHeadIndex() );

        // Collect valid enemy units.
        std::vector<const Unit *> enemies;
        const Force & ownForce = ( unit.GetArmyColor() == arena.getAttackingForce().GetColor() ) ? arena.getAttackingForce() : arena.getDefendingForce();
        const Force & enemyForce = ( &ownForce == &arena.getAttackingForce() ) ? arena.getDefendingForce() : arena.getAttackingForce();

        for ( const Unit * enemy : enemyForce ) {
            if ( enemy != nullptr && enemy->isValid() ) {
                enemies.push_back( enemy );
            }
        }

        // Every candidate below is filtered through the engine's own command validation
        // (Arena::isValid*Command): the geometry-based candidates over-approximate (e.g. cells
        // that are not the head of a reachable wide-unit position, melee attacks of non-blocked
        // archers, shots of blocked archers, moat cells), and a rejected command is silently
        // dropped by ApplyAction*() in Release builds (assert in Debug).

        // Ranged attack from the current position.
        if ( unit.isArchers() && unit.GetShots() > 0 ) {
            for ( const Unit * enemy : enemies ) {
                if ( Arena::isValidAttackCommand( unit, *enemy, -1, -1, 0 ) ) {
                    moves.emplace_back( Command::ATTACK, uid, enemy->GetUID(), -1, -1, 0 );
                }
            }
        }

        // Melee attacks and moves.
        std::set<std::vector<int>> seenAttacks;
        for ( const int32_t cellIdx : cells ) {
            if ( cellIdx != unit.GetHeadIndex() && Arena::isValidMoveCommand( unit, cellIdx ) ) {
                moves.emplace_back( Command::MOVE, uid, cellIdx );
            }

            // Attack from the current position is marked with -1 in the move slot.
            const int32_t attackFrom = ( cellIdx == unit.GetHeadIndex() ? -1 : cellIdx );

            // The cells the unit occupies with its head on cellIdx: a wide unit may also strike
            // from its tail cell (the built-in AI does; the engine accepts it).
            std::vector<int32_t> attackerCells{ cellIdx };
            if ( unit.isWide() ) {
                const Position position = ( attackFrom == -1 ? unit.GetPosition() : Position::GetReachable( unit, cellIdx ) );
                if ( position.GetTail() != nullptr ) {
                    attackerCells.push_back( position.GetTail()->GetIndex() );
                }
            }

            for ( const Unit * enemy : enemies ) {
                // Check all cells occupied by the enemy (a wide unit occupies two cells).
                std::vector<int32_t> enemyCells{ enemy->GetHeadIndex() };
                if ( enemy->isWide() ) {
                    enemyCells.push_back( enemy->GetTailIndex() );
                }

                for ( const int32_t attackerCell : attackerCells ) {
                    for ( const int32_t enemyCell : enemyCells ) {
                        for ( const CellDirection dir : { CellDirection::TOP_LEFT, CellDirection::TOP_RIGHT, CellDirection::RIGHT, CellDirection::BOTTOM_RIGHT,
                                                          CellDirection::BOTTOM_LEFT, CellDirection::LEFT } ) {
                            const Cell * neighbor = Board::GetCell( attackerCell, dir );
                            if ( neighbor == nullptr || neighbor->GetIndex() != enemyCell ) {
                                continue;
                            }

                            if ( !Arena::isValidAttackCommand( unit, *enemy, attackFrom, enemyCell, static_cast<int>( dir ) ) ) {
                                continue;
                            }

                            const Command attack( Command::ATTACK, uid, enemy->GetUID(), attackFrom, enemyCell, static_cast<int>( dir ) );
                            if ( seenAttacks.insert( std::vector<int>( attack.begin(), attack.end() ) ).second ) {
                                moves.push_back( attack );
                            }
                        }
                    }
                }
            }
        }

        moves.emplace_back( Command::SKIP, uid );

        // Hero spells (appended after the unit's moves: the unit's part of the list is unchanged
        // for battles without a casting commander).
        const std::vector<Command> spells = EnumerateSpellCasts( arena );
        moves.insert( moves.end(), spells.begin(), spells.end() );

        return moves;
    }

    std::vector<Command> EnumerateSpellCasts( const Arena & arena )
    {
        std::vector<Command> casts;

        // The commander of the side to move (a hypnotized/berserk unit acts for its current
        // color, like Arena::ApplyActionSpellCast() uses the current force's commander).
        const HeroBase * commander = arena.GetCurrentCommander();
        if ( commander == nullptr || arena.isDisableCastSpell( Spell( Spell::NONE ) ) ) {
            // No hero, the Sphere of Negation, or a spell was already cast this round.
            return casts;
        }

        // Live units on the board, each once, addressed by its head cell (the cell the built-in
        // AI targets; any cell of a wide unit selects the same unit).
        std::vector<const Unit *> boardUnits;
        for ( int32_t idx = 0; idx < Board::sizeInCells; ++idx ) {
            const Unit * unit = Board::GetCell( idx )->GetUnit();
            if ( unit != nullptr && unit->GetHeadIndex() == idx ) {
                boardUnits.push_back( unit );
            }
        }

        std::set<int> seenSpells;
        for ( const Spell & spell : commander->getAllSpells() ) {
            // The same checks as the spell book of the battle interface and the engine.
            if ( !spell.isCombat() || !seenSpells.insert( spell.GetID() ).second || arena.isDisableCastSpell( spell ) || !commander->CanCastSpell( spell ) ) {
                continue;
            }

            const int spellId = spell.GetID();

            if ( spell.isApplyWithoutFocusObject() ) {
                // Mass spells, summoning, Armageddon, Earthquake, ...: no target.
                casts.emplace_back( Command::SPELLCAST, spellId, -1 );
                continue;
            }

            if ( !spell.isApplyToFriends() && !spell.isApplyToEnemies() && !spell.isApplyToAnyTroops() ) {
                // Area spells (Fireball, Meteor Shower, ...) may target any cell.
                for ( int32_t idx = 0; idx < Board::sizeInCells; ++idx ) {
                    casts.emplace_back( Command::SPELLCAST, spellId, idx );
                }
                continue;
            }

            if ( spellId == Spell::TELEPORT ) {
                for ( const Unit * unit : boardUnits ) {
                    if ( !unit->AllowApplySpell( spell, commander ) ) {
                        continue;
                    }
                    for ( int32_t dst = 0; dst < Board::sizeInCells; ++dst ) {
                        const Cell * cell = Board::GetCell( dst );
                        if ( cell->GetUnit() == nullptr && cell->isPassableForUnit( *unit ) ) {
                            casts.emplace_back( Command::SPELLCAST, spellId, unit->GetHeadIndex(), dst );
                        }
                    }
                }
                continue;
            }

            // Spells aimed at a unit (Mirror Image too: its argument is the unit's cell).
            for ( const Unit * unit : boardUnits ) {
                if ( unit->AllowApplySpell( spell, commander ) ) {
                    casts.emplace_back( Command::SPELLCAST, spellId, unit->GetHeadIndex() );
                }
            }

            // Resurrection from the graveyard (a cell without a live unit).
            for ( int32_t idx = 0; idx < Board::sizeInCells; ++idx ) {
                if ( Board::GetCell( idx )->GetUnit() == nullptr && arena.isAbleToResurrectFromGraveyard( idx, spell ) ) {
                    casts.emplace_back( Command::SPELLCAST, spellId, idx );
                }
            }
        }

        return casts;
    }

    std::string EncodeCommander( const Army & army )
    {
        // Castle captains are not heroes: they travel with their castle (EncodeCastle()).
        const Heroes * hero = dynamic_cast<const Heroes *>( army.GetCommander() );
        if ( hero == nullptr ) {
            return {};
        }

        RWStreamBuf stream;
        stream << *hero;

        return encodeHex( stream );
    }

    std::string EncodeCastle( const Castle & castle )
    {
        RWStreamBuf stream;
        stream << castle;

        return encodeHex( stream );
    }

    bool RunBattleServer()
    {
        const char * enabled = std::getenv( "FHEROES2_BATTLE_SERVER" );
        if ( enabled == nullptr || *enabled == '\0' ) {
            return false;
        }

        Maps::FileInfo mapInfo;
        if ( !fheroes2::pickPlaytestMap( mapInfo ) ) {
            return true;
        }

        Settings & conf = Settings::Get();
        conf.setCurrentMapInfo( mapInfo );
        conf.GetPlayers().Init( mapInfo );
        conf.GetPlayers().SetStartGame();

        // Real games driven by the agents are autonomous playtests where every player is AI-controlled. Control matters in
        // battles (e.g. the bad-morale draw gives AI units an extra roll), so the maps' human slots become AI here too:
        // otherwise a replicated battle consumes the random stream differently and desyncs.
        for ( Player * player : conf.GetPlayers() ) {
            player->SetControl( CONTROL_AI );
        }

        if ( mapInfo.version == GameVersion::RESURRECTION ) {
            world.loadResurrectionMap( mapInfo.filename );
        }
        else {
            world.LoadMapMP2( mapInfo.filename, ( mapInfo.version == GameVersion::SUCCESSION_WARS ) );
        }

        // World::Defaults() seeds the world randomly in every process; battle obstacles are
        // derived from the world seed, which would make generated datasets and gate runs
        // irreproducible across engine restarts. Pin the seed in battle-server mode.
        world.SetMapSeed( pinnedWorldSeed );

        clearHeroesFromTiles();

        BattleServer server;
        std::string line;

        // Strict request/response protocol: every operation produces exactly one state reply,
        // the engine never blocks mid-protocol (the main line is replayed from the root on
        // every "action" operation).
        while ( !server.isQuitRequested() && std::getline( std::cin, line ) ) {
            if ( line.find( "\"quit\"" ) != std::string::npos ) {
                break;
            }

            if ( line.find( "\"new\"" ) != std::string::npos ) {
                const uint32_t seed = static_cast<uint32_t>( extractInt( line, "seed", 1 ) );
                const int32_t tile = static_cast<int32_t>( extractInt( line, "tile", -1 ) );
                const auto attackingStacks = parseStacks( extractString( line, "att" ) );
                const auto defendingStacks = parseStacks( extractString( line, "def" ) );

                // Real-battle replication (battle_agent.cpp): the client may pass the world seed
                // of the real game so that obstacle placement (derived from the world seed + the
                // battle tile) matches. 0 or absent means the pinned default — never the seed of
                // a previous "new" operation, so replies do not depend on the operation history.
                const uint32_t worldSeed = static_cast<uint32_t>( extractInt( line, "wseed", 0 ) );
                world.SetMapSeed( worldSeed != 0 ? worldSeed : pinnedWorldSeed );

                // Battle formation of the real armies (board positions derive from it).
                const bool attackingSpread = extractInt( line, "sat", 1 ) != 0;
                const bool defendingSpread = extractInt( line, "sdf", 1 ) != 0;

                // Real-battle replication: the commanders (hero id + hex save-game serialization, see
                // EncodeCommander()) and the army colors (PlayerColor values; neutral = 0).
                const CommanderSpec attackingCommander = parseCommander( line, "ahid", "ahero" );
                const CommanderSpec defendingCommander = parseCommander( line, "dhid", "dhero" );
                const int attackingColor = static_cast<int>( extractInt( line, "acol", -1 ) );
                const int defendingColor = static_cast<int>( extractInt( line, "dcol", -1 ) );

                // The castle/town on the battle tile (hex save-game serialization, see EncodeCastle()) and
                // whether its garrison defends. Bad data is an error, not a silently different battle.
                std::vector<uint8_t> castleData;
                const std::string castleHex = extractString( line, "castle" );
                const bool castleOk = castleHex.empty() || ( decodeHex( castleHex, castleData ) && !castleData.empty() );
                const bool defendingGarrison = extractInt( line, "dgar", 0 ) != 0;
                // Handicap: every stack of a side at this percentage of its count (1..10000).
                const int attackingScale = static_cast<int>( std::clamp<int64_t>( extractInt( line, "ascl", 100 ), 1, 10000 ) );
                const int defendingScale = static_cast<int>( std::clamp<int64_t>( extractInt( line, "dscl", 100 ), 1, 10000 ) );

                if ( !castleOk
                     || !server.newBattle( seed, attackingStacks, defendingStacks, tile, attackingSpread, defendingSpread, attackingCommander, defendingCommander,
                                           attackingColor, defendingColor, castleData, defendingGarrison, attackingScale, defendingScale ) ) {
                    std::cout << "{\"ev\":\"error\",\"what\":\"bad battle setup\"}\n";
                    std::cout.flush();
                }
            }
            else if ( line.find( "\"action\"" ) != std::string::npos ) {
                // One more action on the main line: replay the path plus this action.
                const int64_t act = extractInt( line, "act", static_cast<int64_t>( CommandType::SKIP ) );
                const std::vector<int64_t> args = extractIntArray( line, "args" );

                std::vector<int> rawValues;
                rawValues.reserve( args.size() );
                for ( const int64_t value : args ) {
                    rawValues.push_back( static_cast<int>( value ) );
                }

                server.replay( std::vector<Command>{ Command::FromRaw( static_cast<CommandType>( act ), rawValues ) }, true );
            }
            else if ( line.find( "\"reset\"" ) != std::string::npos ) {
                server.resetLine();
            }
            else if ( line.find( "\"auto\"" ) != std::string::npos ) {
                // Play the current battle with the built-in AI, streaming expert records.
                server.runAuto();
            }
            else if ( line.find( "\"replay\"" ) != std::string::npos ) {
                // Batched replay for search: apply the given path from the root without touching
                // the main line.
                // "full":1 forces the replay from the battle root (reference for the snapshot fast path).
                server.replay( parseCommandPath( line ), false, extractInt( line, "full", 0 ) != 0 );
            }
            else if ( line.find( "\"snap\"" ) != std::string::npos ) {
                // Store the current pause-point state under the given id (the reply carries the
                // current state; the client usually already has it).
                server.snapshotSave( static_cast<int32_t>( extractInt( line, "id", 0 ) ) );
            }
            else if ( line.find( "\"restore\"" ) != std::string::npos ) {
                // Rewind to a snapshot; optionally apply an action path suffix from it and save
                // the resulting state under another id — one roundtrip per search-tree node.
                const int32_t id = static_cast<int32_t>( extractInt( line, "id", 0 ) );
                const int32_t saveAs = static_cast<int32_t>( extractInt( line, "save_as", 0 ) );
                server.snapshotRestore( id, saveAs, parseCommandPath( line ), extractInt( line, "rollout", 0 ) != 0 );
            }
            else if ( line.find( "\"snap_free\"" ) != std::string::npos ) {
                // Release all stored snapshots (search done for this battle).
                server.snapshotsClear();
            }
            else if ( line.find( "\"suggest\"" ) != std::string::npos ) {
                // The built-in AI's action for the current unit, without applying it.
                server.suggest();
            }
            // Any other input at the top level is ignored.
        }

        return true;
    }
}
