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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Funds;

enum class PlayerColor : uint8_t;

namespace AILog
{
    // JSON-lines event emitter used to observe AI behavior and to integrate external AI models. One JSON object per
    // line is appended to the file specified by the FHEROES2_AI_LOG environment variable. If this variable is not set
    // or is empty, the emitter is disabled and all operations below are effectively no-ops.
    //
    // Usage:
    //     AILog::Event ev( "hero_target" );
    //     ev.key( "h" ).value( hero.GetID() );
    //     ev.key( "to" ).value( targetIndex );
    // Nested arrays and objects are supported through beginArray()/beginObject()/endArray()/endObject(); all open
    // containers are closed automatically when the Event object is destroyed.
    class Event
    {
    public:
        // The event type is written into the "ev" field.
        explicit Event( const char * type );

        ~Event();

        Event( const Event & ) = delete;
        Event & operator=( const Event & ) = delete;

        bool isEnabled() const
        {
            return _enabled;
        }

        // Sets the key for the next value or container. Every value must be preceded by a key() call.
        Event & key( const char * name );

        Event & value( const char * text )
        {
            return writeString( text );
        }

        Event & value( const std::string & text )
        {
            return writeString( text.c_str() );
        }

        Event & value( int32_t number )
        {
            return writeNumber( static_cast<int64_t>( number ) );
        }

        Event & value( int64_t number )
        {
            return writeNumber( number );
        }

        Event & value( uint32_t number )
        {
            return writeNumber( static_cast<int64_t>( number ) );
        }

        Event & value( double number );

        // Writes a one-letter color code: B, G, R, Y, O, P or N for PlayerColor::NONE.
        Event & value( PlayerColor color );

        // Writes the given funds as a positional array: [ wood, mercury, ore, sulfur, crystal, gems, gold ].
        Event & value( const Funds & funds );

        Event & beginArray();
        Event & beginObject();
        Event & endArray();
        Event & endObject();

    private:
        Event & writeString( const char * text );
        Event & writeNumber( int64_t number );

        void writeEscapedString( const char * text );

        // Writes a comma before the next member or element unless it directly follows a key.
        void beforeValue();

        // Marks the current container as containing at least one member or element, so that a comma is written
        // before the next one.
        void markValueWritten();

        // Closes all containers opened inside the root object. Used as a safety net in the destructor.
        void closeAllContainers();

        std::string _buffer;
        // "The container already has members or elements" flag for each open container, the root object included.
        std::vector<bool> _hasElements;
        // Opening brackets of all currently open containers in the opening order, used by closeAllContainers().
        std::vector<char> _openBrackets;
        bool _afterKey = false;
        bool _enabled = false;
    };

    // Assigns a new battle ID. Must be called exactly once at the beginning of each battle.
    uint32_t beginBattle();

    // Returns the ID of the current battle assigned by beginBattle().
    uint32_t currentBattleId();
}
