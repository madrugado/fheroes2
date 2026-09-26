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
#include <vector>

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

    // "13x10,21x24" -> [(13, 10), (21, 24)]
    std::vector<std::pair<int32_t, uint32_t>> parseStacks( const std::string & text )
    {
        std::vector<std::pair<int32_t, uint32_t>> result;

        size_t offset = 0;
        while ( offset < text.size() ) {
            const size_t next = text.find( ',', offset );
            const std::string token = text.substr( offset, ( next == std::string::npos ? text.size() : next ) - offset );
            const size_t sep = token.find( 'x' );
            if ( sep != std::string::npos ) {
                result.emplace_back( std::stoi( token.substr( 0, sep ) ), static_cast<uint32_t>( std::stoi( token.substr( sep + 1 ) ) ) );
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
    // Thrown by the action provider when the client sent a control operation (reset/new/quit)
    // instead of an action; unwinds the battle and returns control to the server's main loop.
    struct AbortBattle
    {};

    // Thrown by the action provider when the replay command queue is exhausted; unwinds the
    // battle and reports the current state to the client.
    struct PauseBattle
    {};

    class BattleServer
    {
    public:
        bool newBattle( const uint32_t seed, const std::vector<std::pair<int32_t, uint32_t>> & attackingStacks,
                        const std::vector<std::pair<int32_t, uint32_t>> & defendingStacks, int32_t tileIndex );
        void resetBattle();

        // Plays the current battle to the end, exchanging actions with the client at every unit
        // activation (see the protocol in az/README.md).
        void play();

        // Resets the battle and applies the given action sequence internally (one roundtrip for
        // the whole path). Used by replay-based search. Reports the state at the pause point or
        // the final result.
        void replay( const std::vector<Command> & actionQueue );

        bool isQuitRequested() const
        {
            return _quitRequested;
        }

        // A control operation ("reset"/"new"/"quit") received while the battle was waiting for
        // an action; the main loop must process it after the current battle unwinds.
        const std::string & pendingLine() const
        {
            return _pendingLine;
        }

        void clearPendingLine()
        {
            _pendingLine.clear();
        }

    private:
        // Called by Arena::UnitTurn() at every unit activation: reports the current state and
        // legal moves, then blocks until the client replies with an action.
        bool requestAction( Actions & actions );

        std::string serializeState( const Unit * currentUnit, const std::vector<Command> & legalMoves ) const;
        std::vector<Command> enumerateLegalMoves( const Unit & unit ) const;

        void emitResult();

        Army _attackingArmy;
        Army _defendingArmy;
        std::unique_ptr<Rand::PCG32> _randomGenerator;
        std::unique_ptr<Arena> _arena;

        // Setup of the current battle, used by resetBattle() for replay-based search.
        uint32_t _seed = 0;
        int32_t _tileIndex = -1;
        std::vector<std::pair<int32_t, uint32_t>> _attackingStacks;
        std::vector<std::pair<int32_t, uint32_t>> _defendingStacks;

        bool _quitRequested = false;
        std::string _pendingLine;
    };

    bool BattleServer::newBattle( const uint32_t seed, const std::vector<std::pair<int32_t, uint32_t>> & attackingStacks,
                                  const std::vector<std::pair<int32_t, uint32_t>> & defendingStacks, int32_t tileIndex )
    {
        _seed = seed;
        _attackingStacks = attackingStacks;
        _defendingStacks = defendingStacks;
        _quitRequested = false;

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

        resetBattle();

        return true;
    }

    void BattleServer::resetBattle()
    {
        _attackingArmy.Reset();
        _defendingArmy.Reset();

        for ( const auto & [mon, qty] : _attackingStacks ) {
            _attackingArmy.AssignToFirstFreeSlot( Troop( Monster( mon ), qty ), qty );
        }

        for ( const auto & [mon, qty] : _defendingStacks ) {
            _defendingArmy.AssignToFirstFreeSlot( Troop( Monster( mon ), qty ), qty );
        }

        _attackingArmy.SetColor( PlayerColor::RED );
        _defendingArmy.SetColor( PlayerColor::BLUE );

        // Destroy the old arena first: only one Arena instance may exist at a time (the class
        // keeps a static pointer to the current instance).
        _arena.reset();

        _randomGenerator = std::make_unique<Rand::PCG32>( _seed );
        _arena = std::make_unique<Arena>( _attackingArmy, _defendingArmy, _tileIndex, false, *_randomGenerator );
    }

    void BattleServer::play()
    {
        auto provider = [this]( Actions & actions ) { return requestAction( actions ); };

        try {
            while ( _arena->BattleValid() ) {
                _arena->Turns( provider );
            }
        }
        catch ( const AbortBattle & ) {
            // The client has requested a control operation; the battle is abandoned.
            return;
        }

        // The battle is over: report the final state with the result.
        std::cout << serializeState( nullptr, {} );
        std::cout << "}\n";
        std::cout.flush();
    }

    void BattleServer::replay( const std::vector<Command> & actionQueue )
    {
        VERBOSE_LOG( "replay: reset, " << actionQueue.size() << " actions" )
        resetBattle();
        VERBOSE_LOG( "replay: reset done" )

        std::vector<Command> queue = actionQueue;

        auto provider = [&queue]( Actions & actions ) {
            if ( queue.empty() ) {
                throw PauseBattle{};
            }

            actions.push_back( queue.front() );
            queue.erase( queue.begin() );

            return true;
        };

        try {
            while ( _arena->BattleValid() ) {
                _arena->Turns( provider );
            }
        }
        catch ( const PauseBattle & ) {
            VERBOSE_LOG( "replay: pause, " << queue.size() << " actions left" )
            // The queue is exhausted: report the state at the pause point (with legal moves).
            const Unit * unit = _arena->getCurrentUnit();
            std::cout << serializeState( unit, ( unit != nullptr ? enumerateLegalMoves( *unit ) : std::vector<Command>{} ) );
            std::cout << "}\n";
            std::cout.flush();
            return;
        }
        catch ( const AbortBattle & ) {
            return;
        }

        // The battle is over: report the final state with the result.
        std::cout << serializeState( nullptr, {} );
        std::cout << "}\n";
        std::cout.flush();
    }

    void BattleServer::emitResult() {}

    bool BattleServer::requestAction( Actions & actions )
    {
        const Unit * unit = _arena->getCurrentUnit();

        if ( unit == nullptr ) {
            return false;
        }

        const std::vector<Command> legalMoves = enumerateLegalMoves( *unit );

        {
            std::string state = serializeState( unit, legalMoves );
            std::cout << state << "}\n";
            std::cout.flush();
        }

        std::string line;
        while ( std::getline( std::cin, line ) ) {
            if ( line.find( "\"action\"" ) != std::string::npos ) {
                const int64_t act = extractInt( line, "act", static_cast<int64_t>( CommandType::SKIP ) );
                const std::vector<int64_t> args = extractIntArray( line, "args" );

                std::vector<int> rawValues;
                rawValues.reserve( args.size() );
                for ( const int64_t value : args ) {
                    rawValues.push_back( static_cast<int>( value ) );
                }

                actions.push_back( Command::FromRaw( static_cast<CommandType>( act ), rawValues ) );
                return true;
            }
            if ( line.find( "\"reset\"" ) != std::string::npos || line.find( "\"new\"" ) != std::string::npos || line.find( "\"quit\"" ) != std::string::npos
                 || line.find( "\"replay\"" ) != std::string::npos ) {
                _pendingLine = line;
                if ( line.find( "\"quit\"" ) != std::string::npos ) {
                    _quitRequested = true;
                }

                throw AbortBattle{};
            }
            // Ignore empty or unknown lines and keep waiting for a proper operation.
        }

        // The client is gone.
        _quitRequested = true;
        throw AbortBattle{};
    }

    std::string BattleServer::serializeState( const Unit * currentUnit, const std::vector<Command> & legalMoves ) const
    {
        std::ostringstream out;
        out << "{\"ev\":\"state\",\"turn\":" << _arena->GetTurnNumber()
            << ",\"cur\":" << ( currentUnit != nullptr ? static_cast<int64_t>( currentUnit->GetUID() ) : -1 );

        out << ",\"units\":[";
        bool firstUnit = true;
        for ( const int side : { 0, 1 } ) {
            const Force & force = ( side == 0 ) ? _arena->getAttackingForce() : _arena->getDefendingForce();
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

        if ( !_arena->BattleValid() ) {
            const Result & result = _arena->GetResult();
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

    std::vector<Command> BattleServer::enumerateLegalMoves( const Unit & unit ) const
    {
        std::vector<Command> moves;

        const uint32_t uid = unit.GetUID();

        // All cells reachable by the unit's head on the current turn, plus the current position.
        std::vector<int32_t> cells = _arena->getAllAvailableMoves( unit );
        cells.push_back( unit.GetHeadIndex() );

        // Collect valid enemy units.
        std::vector<const Unit *> enemies;
        const Force & ownForce = ( unit.GetArmyColor() == _arena->getAttackingForce().GetColor() ) ? _arena->getAttackingForce() : _arena->getDefendingForce();
        const Force & enemyForce = ( &ownForce == &_arena->getAttackingForce() ) ? _arena->getDefendingForce() : _arena->getAttackingForce();

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

        BattleServer server;
        std::string line;
        std::string pendingLine;

        while ( !server.isQuitRequested() ) {
            if ( !pendingLine.empty() ) {
                // A control operation arrived while the battle was waiting for an action.
                line = pendingLine;
                pendingLine.clear();
            }
            else if ( !std::getline( std::cin, line ) ) {
                break;
            }

            server.clearPendingLine();

            if ( line.find( "\"quit\"" ) != std::string::npos ) {
                break;
            }

            if ( line.find( "\"new\"" ) != std::string::npos ) {
                const uint32_t seed = static_cast<uint32_t>( extractInt( line, "seed", 1 ) );
                const int32_t tile = static_cast<int32_t>( extractInt( line, "tile", -1 ) );
                const auto attackingStacks = parseStacks( extractString( line, "att" ) );
                const auto defendingStacks = parseStacks( extractString( line, "def" ) );

                if ( !server.newBattle( seed, attackingStacks, defendingStacks, tile ) ) {
                    std::cout << "{\"ev\":\"error\",\"what\":\"no tile\"}\n";
                    std::cout.flush();
                    continue;
                }

                server.play();
            }
            else if ( line.find( "\"reset\"" ) != std::string::npos ) {
                server.resetBattle();
                server.play();
            }
            else if ( line.find( "\"replay\"" ) != std::string::npos ) {
                // Batched replay: reset + apply the whole action path inside the engine.
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

                server.replay( queue );
            }
            // Any other input at the top level is ignored.

            pendingLine = server.pendingLine();
        }

        return true;
    }
}
