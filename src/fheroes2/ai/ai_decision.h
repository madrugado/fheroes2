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

class Castle;
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

    // Replies of the choice requests below that are not a candidate index.
    constexpr int32_t replySkip = -2; // the built-in AI decides
    constexpr int32_t replyNone = -1; // the agent explicitly chose to do nothing

    struct BuildCandidate
    {
        uint32_t building = 0;
        // The kingdom lacks some resources, but a marketplace trade can cover them (the building
        // is then bought via AI::BuildIfPossible(), which trades first).
        bool needsTrade = false;
    };

    // Asks the external agent what to build in the castle this turn (at most one building per
    // castle and day). Returns the index of the chosen candidate, replyNone (build nothing and
    // save the resources) or replySkip (built-in castle development).
    int32_t requestBuild( const Castle & castle, const std::vector<BuildCandidate> & candidates, const bool defensive );

    // Reports what was built in the castle during its development step (0 = nothing), and
    // whether the external agent or the built-in AI decided.
    void reportBuildResult( const Castle & castle, const uint32_t building, const bool byAgent );

    struct HireCandidate
    {
        Castle * castle = nullptr;
        // 1 or 2: which of the two heroes offered in the kingdom's taverns.
        int slot = 1;
        Heroes * hero = nullptr;
    };

    // Asks the external agent whether and where to hire a hero. `builtinChoice` is the index of
    // the candidate the built-in AI would hire (replyNone if it would not hire). Returns the
    // index of the chosen candidate, replyNone or replySkip.
    int32_t requestHire( const Kingdom & kingdom, const std::vector<HireCandidate> & candidates, const int32_t builtinChoice );

    // Reports that a playthrough has ended (autonomous playtest mode). The caller checks that an
    // external agent channel (strategic or battle) is enabled: both agents consume "game_end".
    void sendGameOver( const uint32_t playthroughId, const char * summaryJson );
}
