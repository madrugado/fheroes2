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

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ai_battle.h"
#include "army.h"
#include "army_troop.h"
#include "battle.h"
#include "battle_arena.h"
#include "battle_army.h"
#include "battle_cell.h"
#include "battle_command.h"
#include "battle_board.h"
#include "battle_troop.h"
#include "color.h"
#include "game_auto_playtest.h"
#include "maps_fileinfo.h"
#include "logging.h"
#include "maps.h"
#include "maps_tiles.h"
#include "monster.h"
#include "mp2.h"
#include "players.h"
#include "rand.h"
#include "settings.h"
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

    // "13x10,21x24" or with explicit army slots "0:13x10,2:21x24" -> StackSpec list.
    std::vector<StackSpec> parseStacks( const std::string & text )
    {
        std::vector<StackSpec> result;

        size_t offset = 0;
        while ( offset < text.size() ) {
            const size_t next = text.find( ',', offset );
            const std::string token = text.substr( offset, ( next == std::string::npos ? text.size() : next ) - offset );

            StackSpec spec;

            std::string_view rest{ token };
            const size_t slotSep = token.find( ':' );
            if ( slotSep != std::string::npos ) {
                spec.slot = std::stoi( token.substr( 0, slotSep ) );
                rest = std::string_view{ token }.substr( slotSep + 1 );
            }

            const size_t sep = rest.find( 'x' );
            if ( sep != std::string::npos ) {
                spec.mon = std::stoi( std::string( rest.substr( 0, sep ) ) );
                spec.count = static_cast<uint32_t>( std::stoi( std::string( rest.substr( sep + 1 ) ) ) );
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

    class BattleServer
    {
    public:
        bool newBattle( const uint32_t seed, const std::vector<StackSpec> & attackingStacks, const std::vector<StackSpec> & defendingStacks, int32_t tileIndex,
                        const bool attackingSpreadFormation, const bool defendingSpreadFormation );
        void resetBattle();

        // Plays the current battle to the end, exchanging actions with the client at every unit
        // activation (see the protocol in az/README.md).
        void play();

        // Applies the given action sequence from the battle root inside the engine (one
        // roundtrip for the whole path). With extendPath the sequence is appended to the main
        // line (the "action" operation); otherwise the main line is untouched (MCTS replays).
        void replay( const std::vector<Command> & actionQueue, const bool extendPath );

        // Resets the main line to the battle root and reports the root state.
        void resetLine();

        // Plays the current battle to the end with the built-in battle AI, streaming one expert
        // record per decision: the full state (with legal moves) plus the action chosen by the
        // built-in AI. Used to generate training data from the ready-made algorithms.
        void runAuto();

        // Applies the given actions from the current battle root, pausing at the next decision
        // point (or when the battle ends). With resumeCurrentRound the battle is resumed where
        // a snapshot restore left it (mid-round) instead of starting a fresh turn.
        void advance( const std::vector<Command> & path, const bool resumeCurrentRound = false );

        // Snapshot/restore of the battle state for the search tree: "snap" stores the current
        // pause-point state under a client-chosen id, "restore" rewinds to it (optionally
        // applying an action path suffix and saving the result under another id) in one
        // roundtrip. Snapshots live inside the Arena and die with it ("new"/"reset").
        void snapshotSave( const int32_t id );
        void snapshotRestore( const int32_t id, const int32_t saveAsId, const std::vector<Command> & path );
        void snapshotsClear();

        // Reports the current state with the action the built-in battle AI would take for the
        // unit to move (in the "expert" field); the action is not applied. Lets an external
        // agent play against the built-in AI in the gate runner (az/gate.py).
        void suggest();

        // Reports the current state: a decision point (with legal moves) or the final result.
        void emitState();

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

        // Setup of the current battle, used by resetBattle() for replay-based search.
        uint32_t _seed = 0;
        int32_t _tileIndex = -1;
        std::vector<StackSpec> _attackingStacks;
        std::vector<StackSpec> _defendingStacks;
        bool _attackingSpreadFormation = true;
        bool _defendingSpreadFormation = true;

        bool _quitRequested = false;

        // Actions applied since the battle root; the main line is replayed from scratch on
        // every "action" operation (cheap for the engine, keeps the protocol stateless).
        std::vector<Command> _currentPath;
    };

    bool BattleServer::newBattle( const uint32_t seed, const std::vector<StackSpec> & attackingStacks, const std::vector<StackSpec> & defendingStacks,
                                  int32_t tileIndex, const bool attackingSpreadFormation, const bool defendingSpreadFormation )
    {
        _seed = seed;
        _attackingStacks = attackingStacks;
        _defendingStacks = defendingStacks;
        _attackingSpreadFormation = attackingSpreadFormation;
        _defendingSpreadFormation = defendingSpreadFormation;
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

        resetBattle();
        advance( {} );
        emitState();

        return true;
    }

    void BattleServer::resetBattle()
    {
        _attackingArmy.Reset();
        _defendingArmy.Reset();

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

        _attackingArmy.SetColor( PlayerColor::RED );
        _defendingArmy.SetColor( PlayerColor::BLUE );
        _attackingArmy.SetSpreadFormation( _attackingSpreadFormation );
        _defendingArmy.SetSpreadFormation( _defendingSpreadFormation );

        // Destroy the old arena first: only one Arena instance may exist at a time (the class
        // keeps a static pointer to the current instance).
        _arena.reset();

        _randomGenerator = std::make_unique<Rand::PCG32>( _seed );
        _arena = std::make_unique<Arena>( _attackingArmy, _defendingArmy, _tileIndex, false, *_randomGenerator );
    }

    void BattleServer::resetLine()
    {
        _currentPath.clear();
        resetBattle();
        advance( {} );
        emitState();
    }

    void BattleServer::advance( const std::vector<Command> & path, const bool resumeCurrentRound )
    {
        std::vector<Command> queue = path;

        // The provider feeds the queued actions to the engine; when the queue is exhausted the
        // battle is paused at the next decision point (PauseBattle unwinds the simulation).
        auto provider = [&queue]( Actions & actions ) {
            if ( queue.empty() ) {
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

    void BattleServer::snapshotRestore( const int32_t id, const int32_t saveAsId, const std::vector<Command> & path )
    {
        const auto snapshotIter = _snapshots.find( id );
        if ( snapshotIter == _snapshots.end() || !_arena->applySnapshot( *snapshotIter->second ) ) {
            std::cout << "{\"ev\":\"error\",\"what\":\"unknown snapshot id\"}\n";
            std::cout.flush();
            return;
        }

        advance( path, true );

        if ( saveAsId > 0 ) {
            _snapshots[saveAsId] = _arena->captureSnapshot();
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
        std::cout << SerializeArenaState( *_arena, unit, ( unit != nullptr ? EnumerateLegalMoves( *_arena, *unit ) : std::vector<Command>{} ) );

        if ( unit != nullptr ) {
            // Ask the built-in battle AI exactly like Arena::UnitTurn() does for AI-controlled
            // units (and like runAuto() does), but do not apply the result: the client decides.
            Actions chosen;
            AI::BattlePlanner::Get().BattleTurn( *_arena, *unit, chosen );

            if ( !chosen.empty() ) {
                std::cout << ",\"expert\":{\"act\":" << static_cast<int>( chosen.front().GetType() ) << ",\"args\":[";
                for ( size_t i = 0; i < chosen.front().size(); ++i ) {
                    if ( i > 0 ) {
                        std::cout << ',';
                    }
                    std::cout << chosen.front()[i];
                }
                std::cout << "]}";
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

            // Ask the built-in battle AI exactly like Arena::UnitTurn() does for AI-controlled units.
            Actions chosen;
            AI::BattlePlanner::Get().BattleTurn( *_arena, *unit, chosen );
            if ( chosen.empty() ) {
                return false;
            }

            // Expert record: the pre-decision state (with legal moves) + the built-in AI's action.
            std::cout << SerializeArenaState( *_arena, unit, legalMoves );
            std::cout << ",\"expert\":{\"act\":" << static_cast<int>( chosen.front().GetType() ) << ",\"args\":[";
            for ( size_t i = 0; i < chosen.front().size(); ++i ) {
                if ( i > 0 ) {
                    std::cout << ',';
                }
                std::cout << chosen.front()[i];
            }
            std::cout << "]}}\n";
            std::cout.flush();

            for ( const Command & cmd : chosen ) {
                actions.push_back( cmd );
            }

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

    void BattleServer::replay( const std::vector<Command> & actionQueue, const bool extendPath )
    {
        // The battle is deterministic, so any state materializes by rebuilding the arena at the
        // battle root and applying the whole action path inside the engine (one roundtrip).
        // - extendPath (the "action" operation): the queue extends the main line, which is then
        //   replayed and becomes the new main line.
        // - otherwise (the batched search "replay"): the main line is untouched; the queue is
        //   applied on top of it, materializing the state the search asked for.
        if ( extendPath ) {
            _currentPath.insert( _currentPath.end(), actionQueue.begin(), actionQueue.end() );

            resetBattle();
            advance( _currentPath );
        }
        else {
            std::vector<Command> path = _currentPath;
            path.insert( path.end(), actionQueue.begin(), actionQueue.end() );

            resetBattle();
            advance( path );
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

        // Ranged attack from the current position (the engine applies the melee penalty itself
        // when an enemy is adjacent).
        if ( unit.isArchers() && unit.GetShots() > 0 ) {
            for ( const Unit * enemy : enemies ) {
                moves.emplace_back( Command::ATTACK, uid, enemy->GetUID(), -1, -1, 0 );
            }
        }

        // Melee attacks and moves.
        for ( const int32_t cellIdx : cells ) {
            if ( cellIdx != unit.GetHeadIndex() ) {
                moves.emplace_back( Command::MOVE, uid, cellIdx );
            }

            for ( const Unit * enemy : enemies ) {
                // Check all cells occupied by the enemy (a wide unit occupies two cells).
                std::vector<int32_t> enemyCells{ enemy->GetHeadIndex() };
                if ( enemy->isWide() ) {
                    enemyCells.push_back( enemy->GetTailIndex() );
                }

                for ( const int32_t enemyCell : enemyCells ) {
                    for ( const CellDirection dir : { CellDirection::TOP_LEFT, CellDirection::TOP_RIGHT, CellDirection::RIGHT, CellDirection::BOTTOM_RIGHT,
                                                      CellDirection::BOTTOM_LEFT, CellDirection::LEFT } ) {
                        const Cell * neighbor = Board::GetCell( cellIdx, dir );
                        if ( neighbor == nullptr || neighbor->GetIndex() != enemyCell ) {
                            continue;
                        }

                        // Attack from the current position is marked with -1 in the move slot.
                        moves.emplace_back( Command::ATTACK, uid, enemy->GetUID(), ( cellIdx == unit.GetHeadIndex() ? -1 : cellIdx ), enemyCell,
                                            static_cast<int>( dir ) );
                    }
                }
            }
        }

        moves.emplace_back( Command::SKIP, uid );

        return moves;
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

        if ( mapInfo.version == GameVersion::RESURRECTION ) {
            world.loadResurrectionMap( mapInfo.filename );
        }
        else {
            world.LoadMapMP2( mapInfo.filename, ( mapInfo.version == GameVersion::SUCCESSION_WARS ) );
        }

        // World::Defaults() seeds the world randomly in every process; battle obstacles are
        // derived from the world seed, which would make generated datasets and gate runs
        // irreproducible across engine restarts. Pin the seed in battle-server mode.
        world.SetMapSeed( 20260926 );

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
                // battle tile) matches. 0 keeps the pinned default set at startup.
                const uint32_t worldSeed = static_cast<uint32_t>( extractInt( line, "wseed", 0 ) );
                if ( worldSeed != 0 ) {
                    world.SetMapSeed( worldSeed );
                }

                // Battle formation of the real armies (board positions derive from it).
                const bool attackingSpread = extractInt( line, "sat", 1 ) != 0;
                const bool defendingSpread = extractInt( line, "sdf", 1 ) != 0;

                if ( !server.newBattle( seed, attackingStacks, defendingStacks, tile, attackingSpread, defendingSpread ) ) {
                    std::cout << "{\"ev\":\"error\",\"what\":\"no tile\"}\n";
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
                server.replay( parseCommandPath( line ), false );
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
                server.snapshotRestore( id, saveAs, parseCommandPath( line ) );
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
