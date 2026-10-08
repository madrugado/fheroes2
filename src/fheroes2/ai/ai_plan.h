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

class Army;
class Heroes;
struct VecHeroes;
enum class PlayerColor : uint8_t;

// Whole-game plans for the AI (strategic experiments, rl/strategy_policies.py RulePolicy and
// play_vs_builtin.py --plan). FHEROES2_PLAN="color=Blue,champion=1,secondary_min=1": comma-separated
// key=value; "color" limits the plan to one player (absent = every AI player). Unset = off, the AI
// behaves exactly as upstream.
//
//   champion=1      — one main hero at any number of heroes (upstream assigns a Champion only with
//                     more than three heroes): the strongest army, kept as long as the hero lives;
//                     every other hero is a Courier (brings troops to the champion, collects resources
//                     and dwellings when there is nothing to bring);
//   secondary_min=1 — a non-champion hero keeps a minimal army: when it meets the champion or visits
//                     an own castle it hands over everything except one monster of a fast but weak
//                     kind, and it takes no troops from castle garrisons; the champion values visits of
//                     his own castles fully (he must come back for those troops);
//                     secondary_min=2 — only on meeting the champion: in a castle a secondary hero takes the
//                     garrison as a built-in courier does and carries it to the champion;
//   garrison_slowest=1 — the champion, after taking a castle's garrison, leaves his slowest troop there
//                     (the castle is not left empty, the slowest troop limits his movement anyway);
//   champion_skills=1 — the champion's level-up choice values logistics and the fighting secondaries
//                     and never scouting, estates, diplomacy or eagle eye (ai_planner_hero.cpp);
//   secondary_skills=1 — every other hero values estates first, then logistics and pathfinding.
namespace AIPlan
{
    // The value of `key` in the plan for this player (0 = off).
    int value( const PlayerColor color, const char * key );

    // champion=1: assigns the AI roles of the kingdom's heroes. Returns false when the plan is off
    // for this kingdom (the built-in role assignment runs then).
    bool assignRoles( VecHeroes & heroes );

    // secondary_min=1 and `hero` is not the champion.
    bool keepsMinimalArmy( const Heroes & hero );

    // Moves the troops of a hero's army to `receiver` (a hero's army or a castle garrison), keeping one
    // monster of the fastest kind among the weaker half of the hero's monsters.
    void handOverArmy( Army & giver, Army & receiver );
}
