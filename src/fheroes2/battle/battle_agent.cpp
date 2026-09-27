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
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include "battle_agent.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ai_log.h"
#include "army.h"
#include "army_troop.h"
#include "battle.h"
#include "battle_arena.h"
#include "battle_command.h"
#include "battle_server.h"
#include "logging.h"
#include "monster.h"
#include "world.h"

namespace
{
    bool prepareChannel()
    {
        static bool initialized = false;
        static bool enabled = false;

        if ( !initialized ) {
            initialized = true;
            const char * value = std::getenv( "FHEROES2_BATTLE_AGENT" );
            enabled = ( value != nullptr && *value != '\0' );
        }

        return enabled;
    }

    // The protocol channel is broken (the agent is gone or misbehaves): fall back to the
    // built-in battle AI for the rest of the session.
    bool channelBroken = false;

    bool isChannelBroken()
    {
        return channelBroken;
    }

    void markChannelBroken()
    {
        if ( !channelBroken ) {
            channelBroken = true;
            ERROR_LOG( "Battle agent channel is broken: falling back to the built-in battle AI." )
        }
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

    std::vector<int64_t> extractIntArray( const std::string & line, const char * key )
    {
        std::vector<int64_t> result;

        const std::string needle = "\"" + std::string( key ) + "\":[";
        const size_t pos = line.find( needle );
        if ( pos == std::string::npos ) {
            return result;
        }

        const size_t start = pos + needle.size();
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

    void serializeArmy( std::ostringstream & out, const Army & army )
    {
        // The agent reconstructs the battle in its own engine replica: it needs the army slot
        // of every stack (board positions derive from it) and the battle formation.
        out << "{\"spread\":" << ( army.isSpreadFormation() ? 1 : 0 ) << ",\"stacks\":[";
        bool firstStack = true;
        for ( size_t i = 0; i < army.Size(); ++i ) {
            const Troop * troop = army.GetTroop( i );
            if ( troop == nullptr || !troop->isValid() ) {
                continue;
            }

            if ( !firstStack ) {
                out << ',';
            }
            firstStack = false;

            out << '[' << i << ',' << troop->GetID() << ',' << troop->GetCount() << ']';
        }
        out << "]}";
    }

    bool isSameCommand( const Battle::Command & lhs, const Battle::Command & rhs )
    {
        if ( lhs.GetType() != rhs.GetType() || lhs.size() != rhs.size() ) {
            return false;
        }

        for ( size_t i = 0; i < lhs.size(); ++i ) {
            if ( lhs[i] != rhs[i] ) {
                return false;
            }
        }

        return true;
    }
}

bool BattleAgent::isEnabled()
{
    return prepareChannel() && !isChannelBroken();
}

void BattleAgent::battleBegins( const uint32_t seed, const int32_t tileIndex, const Army & attackingArmy, const Army & defendingArmy )
{
    if ( !prepareChannel() ) {
        return;
    }

    std::ostringstream out;
    out << "{\"ev\":\"battle_start\",\"bid\":" << AILog::currentBattleId() << ",\"seed\":" << seed << ",\"tile\":" << tileIndex
        << ",\"wseed\":" << world.GetMapSeed()
        // Siege battles cannot be searched by the agent (no state snapshots for sieges).
        << ",\"searchable\":" << ( Battle::Arena::GetCastle() == nullptr ? 1 : 0 )
        << ",\"att\":";
    serializeArmy( out, attackingArmy );
    out << ",\"def\":";
    serializeArmy( out, defendingArmy );
    out << "}\n";
    std::cout << out.str();
    std::cout.flush();
}

void BattleAgent::battleEnds( const Battle::Result & result )
{
    if ( !isEnabled() ) {
        return;
    }

    const char * winner = "draw";
    if ( result.attacker & Battle::RESULT_WINS ) {
        winner = "att";
    }
    else if ( result.defender & Battle::RESULT_WINS ) {
        winner = "def";
    }

    std::cout << "{\"ev\":\"battle_end\",\"bid\":" << AILog::currentBattleId() << ",\"result\":\"" << winner << "\"}\n";
    std::cout.flush();
}

bool BattleAgent::requestTurn( Battle::Arena & arena, const Battle::Unit & unit, Battle::Actions & actions )
{
    if ( !isEnabled() ) {
        return false;
    }

    const std::vector<Battle::Command> legalMoves = Battle::EnumerateLegalMoves( arena, unit );

    // Decision query: the state format is the battle-server wire format extended with the
    // battle id and the searchability flag; the agent answers with an "action" operation or
    // delegates the decision back to the built-in AI ("planner").
    std::cout << Battle::SerializeArenaState( arena, &unit, legalMoves ) << ",\"bid\":" << AILog::currentBattleId()
              << ",\"searchable\":" << ( Battle::Arena::GetCastle() == nullptr ? 1 : 0 ) << "}\n";
    std::cout.flush();

    std::string line;
    while ( std::getline( std::cin, line ) ) {
        if ( line.find( "\"action\"" ) != std::string::npos ) {
            const int64_t act = extractInt( line, "act", -1 );
            const std::vector<int64_t> rawArgs = extractIntArray( line, "args" );

            std::vector<int> args;
            args.reserve( rawArgs.size() );
            for ( const int64_t value : rawArgs ) {
                args.push_back( static_cast<int>( value ) );
            }

            const Battle::Command command = Battle::Command::FromRaw( static_cast<Battle::CommandType>( act ), args );

            for ( const Battle::Command & legal : legalMoves ) {
                if ( isSameCommand( legal, command ) ) {
                    // The action is logged by the caller (Arena::UnitTurn) together with the
                    // planner's actions.
                    actions.push_back( command );
                    return true;
                }
            }

            // Not a legal move: report and let the built-in AI decide this turn (the channel
            // stays alive, so the agent gets the next decision).
            std::cout << "{\"ev\":\"battle_fallback\",\"bid\":" << AILog::currentBattleId() << ",\"what\":\"invalid action\"}\n";
            std::cout.flush();
            return false;
        }
        if ( line.find( "\"planner\"" ) != std::string::npos || line.find( "\"skip\"" ) != std::string::npos ) {
            // The agent delegated this decision to the built-in AI.
            return false;
        }
        // Ignore unknown lines and keep waiting for a proper operation.
    }

    // The agent is gone.
    markChannelBroken();

    return false;
}
