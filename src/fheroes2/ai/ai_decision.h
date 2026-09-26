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
#include <vector>

class Heroes;
class Kingdom;

namespace AI
{
    struct TargetCandidate;
}

// Decision protocol for external strategic agents (see az/README.md). Enabled by the
// FHEROES2_STRATEGY_SERVER environment variable: at every strategic decision point the
// engine reports its observation and blocks until the external agent replies with a
// decision. If the agent is gone or replies with garbage, the built-in AI takes over.
namespace AIDecision
{
    bool isEnabled();

    // Reports the kingdom-level context (resources, castles, heroes) at the beginning of
    // each AI turn.
    void sendTurnContext( const Kingdom & kingdom );

    // Asks the external agent to choose a target for the given hero. Returns the chosen tile
    // index, or -1 if the agent asked to skip the decision (the built-in AI decides then).
    int32_t requestHeroTarget( const Heroes & hero, const std::vector<AI::TargetCandidate> & candidates );

    // Reports that a playthrough has ended (autonomous playtest mode).
    void sendGameOver( const uint32_t playthroughId, const char * summaryJson );
}
