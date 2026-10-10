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

#pragma once

#include <cstdint>

class Army;

namespace Battle
{
    class Actions;
    class Arena;
    struct Result;
    class Unit;
}

// External battle agent protocol (see rl/README.md): in real battles, AI-controlled units
// ask an external process (the AlphaZero-style battle agent, rl/battle_agent.py) for their
// actions over the JSON-lines channel on stdin/stdout (the same channel the strategic
// protocol uses). Enabled by the FHEROES2_BATTLE_AGENT environment variable.
//
// The channel is self-healing like the strategic protocol: if the agent is gone or replies
// with garbage, the built-in battle AI takes over permanently. An action that is not in the
// enumerated legal list only falls back for the current decision (a "battle_fallback" event
// is reported to the agent).
namespace BattleAgent
{
    bool isEnabled();

    // Reports the battle setup (seed, tile, world seed, army stacks with their army slots and
    // formations) to the external agent right after the arena is constructed. The agent uses
    // it to reconstruct the battle in its own headless engine replica for the tree search.
    void battleBegins( const uint32_t seed, const int32_t tileIndex, const Army & attackingArmy, const Army & defendingArmy );

    // Reports that the battle has ended.
    void battleEnds( const Battle::Result & result );

    // Asks the external agent for the action of the unit to move. Returns true when the agent
    // provided (and the caller should apply) an action; returns false when the built-in battle
    // AI must decide (agent disabled, gone, or delegated the decision).
    bool requestTurn( Battle::Arena & arena, const Battle::Unit & unit, Battle::Actions & actions );

    // Reports the commands the built-in battle AI chose after requestTurn() returned false ("planner_actions",
    // no reply expected): the agent keeps the whole battle history, its own actions and these.
    void reportPlannerActions( const Battle::Actions & actions );
}
