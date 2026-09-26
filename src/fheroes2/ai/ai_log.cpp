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

#include "ai_log.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>

#if defined( _WIN32 )
#include <process.h>
#else
#include <unistd.h>
#endif

#include "color.h"
#include "resource.h"
#include "system.h"

namespace
{
    struct LogFileHolder
    {
        LogFileHolder() = default;

        LogFileHolder( const LogFileHolder & ) = delete;

        ~LogFileHolder()
        {
            if ( file != nullptr ) {
                std::fclose( file );
            }
        }

        LogFileHolder & operator=( const LogFileHolder & ) = delete;

        std::FILE * file = nullptr;
    };

    LogFileHolder & getLogFileHolder()
    {
        static LogFileHolder holder;
        return holder;
    }

    // This mutex protects the log file from interleaved writes in case events are emitted from different threads.
    std::mutex & getLogMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    bool prepareFile()
    {
        static bool initializationAttempted = false;

        LogFileHolder & holder = getLogFileHolder();
        if ( holder.file != nullptr ) {
            return true;
        }

        if ( initializationAttempted ) {
            return false;
        }
        initializationAttempted = true;

        const char * path = std::getenv( "FHEROES2_AI_LOG" );
        if ( path == nullptr || *path == '\0' ) {
            return false;
        }

        holder.file = std::fopen( path, "a" );
        if ( holder.file == nullptr ) {
            return false;
        }

        // The session header is written directly to avoid any dependency on the Event class at this point.
        const tm timeInfo = System::GetTM( std::time( nullptr ) );
        char timeBuffer[64] = {};
        if ( std::strftime( timeBuffer, sizeof( timeBuffer ), "%d.%m.%Y %H:%M:%S", &timeInfo ) == 0 ) {
            timeBuffer[0] = '\0';
        }

#if defined( _WIN32 )
        const long long processId = _getpid();
#else
        const long long processId = static_cast<long long>( getpid() );
#endif

        std::fprintf( holder.file, "{\"ev\":\"session_start\",\"ts\":\"%s\",\"pid\":%lld}\n", timeBuffer, processId );
        std::fflush( holder.file );

        return true;
    }
}

namespace AILog
{
    namespace
    {
        uint32_t battleId = 0;
    }

    uint32_t beginBattle()
    {
        ++battleId;
        return battleId;
    }

    uint32_t currentBattleId()
    {
        return battleId;
    }

    Event::Event( const char * type )
        : _enabled( prepareFile() )
    {
        if ( !_enabled ) {
            return;
        }

        getLogMutex().lock();

        _buffer.reserve( 256 );
        _buffer += "{\"ev\":\"";
        _buffer += type;
        _buffer += "\"";

        // The root object is already open and already has a member, so a comma is needed before the next key.
        _hasElements.push_back( true );
    }

    Event::~Event()
    {
        if ( !_enabled ) {
            return;
        }

        closeAllContainers();

        _buffer += "}\n";

        std::FILE * file = getLogFileHolder().file;
        if ( file != nullptr ) {
            std::fputs( _buffer.c_str(), file );
            std::fflush( file );
        }

        getLogMutex().unlock();
    }

    Event & Event::key( const char * name )
    {
        if ( !_enabled ) {
            return *this;
        }

        beforeValue();

        writeEscapedString( name );
        _buffer += ':';
        _afterKey = true;

        return *this;
    }

    Event & Event::writeString( const char * text )
    {
        if ( !_enabled ) {
            return *this;
        }

        beforeValue();
        writeEscapedString( text );
        markValueWritten();

        return *this;
    }

    Event & Event::writeNumber( int64_t number )
    {
        if ( !_enabled ) {
            return *this;
        }

        beforeValue();
        _buffer += std::to_string( number );
        markValueWritten();

        return *this;
    }

    Event & Event::value( const double number )
    {
        if ( !_enabled ) {
            return *this;
        }

        beforeValue();

        if ( std::isnan( number ) || std::isinf( number ) ) {
            _buffer += "null";
        }
        else {
            char buffer[64] = {};
            std::snprintf( buffer, sizeof( buffer ), "%.2f", number );
            _buffer += buffer;
        }

        markValueWritten();

        return *this;
    }

    Event & Event::value( const PlayerColor color )
    {
        if ( !_enabled ) {
            return *this;
        }

        beforeValue();

        switch ( color ) {
        case PlayerColor::BLUE:
            _buffer += "\"B\"";
            break;
        case PlayerColor::GREEN:
            _buffer += "\"G\"";
            break;
        case PlayerColor::RED:
            _buffer += "\"R\"";
            break;
        case PlayerColor::YELLOW:
            _buffer += "\"Y\"";
            break;
        case PlayerColor::ORANGE:
            _buffer += "\"O\"";
            break;
        case PlayerColor::PURPLE:
            _buffer += "\"P\"";
            break;
        default:
            _buffer += "\"N\"";
            break;
        }

        markValueWritten();

        return *this;
    }

    Event & Event::value( const Funds & funds )
    {
        if ( !_enabled ) {
            return *this;
        }

        beforeValue();

        _buffer += '[';
        _buffer += std::to_string( funds.wood );
        _buffer += ',';
        _buffer += std::to_string( funds.mercury );
        _buffer += ',';
        _buffer += std::to_string( funds.ore );
        _buffer += ',';
        _buffer += std::to_string( funds.sulfur );
        _buffer += ',';
        _buffer += std::to_string( funds.crystal );
        _buffer += ',';
        _buffer += std::to_string( funds.gems );
        _buffer += ',';
        _buffer += std::to_string( funds.gold );
        _buffer += ']';

        markValueWritten();

        return *this;
    }

    Event & Event::beginArray()
    {
        if ( !_enabled ) {
            return *this;
        }

        beforeValue();

        _buffer += '[';
        _hasElements.push_back( false );
        _openBrackets.push_back( '[' );
        _afterKey = false;

        return *this;
    }

    Event & Event::beginObject()
    {
        if ( !_enabled ) {
            return *this;
        }

        beforeValue();

        _buffer += '{';
        _hasElements.push_back( false );
        _openBrackets.push_back( '{' );
        _afterKey = false;

        return *this;
    }

    Event & Event::endArray()
    {
        if ( !_enabled ) {
            return *this;
        }

        _buffer += ']';
        _hasElements.pop_back();
        _openBrackets.pop_back();
        markValueWritten();

        return *this;
    }

    Event & Event::endObject()
    {
        if ( !_enabled ) {
            return *this;
        }

        _buffer += '}';
        _hasElements.pop_back();
        _openBrackets.pop_back();
        markValueWritten();

        return *this;
    }

    void Event::writeEscapedString( const char * text )
    {
        _buffer += '"';

        for ( const char * symbol = text; *symbol != '\0'; ++symbol ) {
            switch ( *symbol ) {
            case '"':
                _buffer += "\\\"";
                break;
            case '\\':
                _buffer += "\\\\";
                break;
            case '\n':
                _buffer += "\\n";
                break;
            case '\r':
                _buffer += "\\r";
                break;
            case '\t':
                _buffer += "\\t";
                break;
            default:
                if ( static_cast<unsigned char>( *symbol ) < 0x20 ) {
                    char buffer[8] = {};
                    std::snprintf( buffer, sizeof( buffer ), "\\u%04X", static_cast<unsigned char>( *symbol ) );
                    _buffer += buffer;
                }
                else {
                    _buffer += *symbol;
                }
                break;
            }
        }

        _buffer += '"';
    }

    void Event::beforeValue()
    {
        if ( !_afterKey && _hasElements.back() ) {
            _buffer += ',';
        }
    }

    void Event::markValueWritten()
    {
        _hasElements.back() = true;
        _afterKey = false;
    }

    void Event::closeAllContainers()
    {
        while ( !_openBrackets.empty() ) {
            _buffer += ( _openBrackets.back() == '[' ) ? ']' : '}';
            _hasElements.pop_back();
            _openBrackets.pop_back();
        }
    }
}
