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
//                     and dwellings when there is nothing to bring); secondaries=1 instead keeps the
//                     built-in roles for them (fighter when much stronger than the median, else hunter);
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
//   chains=1        — troops travel to the champion as a relay: a courier with cargo goes straight to the
//                     champion when it reaches him this turn, else to the own hero it reaches this turn
//                     that stands clearly closer to the champion; between two secondary heroes the army
//                     goes to the one closer to the champion (ai_planner_hero.cpp, ai_hero_action.cpp).
//   defend_relay=1  — a battle far from the champion: a courier with troops carries them to an own castle under threat
//                     that the champion is more than 10 tiles away from and leaves them in its garrison;
//   mana=1          — every castle builds a Mage Guild early, the champion values a night in an own castle with a
//                     guild by his missing spell points (it restores all of them);
//   collect=1       — the secondary heroes value dwellings, mines, resources and artifacts twice (the troops and
//                     artifacts then travel to the champion: couriers, AIMeeting gives artifacts to the higher role);
//   split_singles=1 — before every battle of the champion the stack of his weakest monsters is split into
//                     single-monster stacks in the free slots (they soak the enemy's retaliation strikes
//                     and draw attacks); merged back after the battle so the slots stay free for new troop
//                     types (upstream does this split only from the Hard difficulty on, for every hero);
//                     split_singles=2 splits the fastest stack weaker than the strongest one instead (rl/split_bench.py:
//                     the weakest stack is usually slower than the champion's main stack, its singles move after it
//                     and cannot take the retaliation first).
namespace AIPlan
{
    // The value of `key` in the plan for this player (0 = off).
    int value( const PlayerColor color, const char * key );

    // Sets a key of the plan for every player (the battle server's single-monster split, setSplit()).
    void setValue( const char * key, const int value );

    // champion=1: assigns the AI roles of the kingdom's heroes. Returns false when the plan is off
    // for this kingdom (the built-in role assignment runs then).
    bool assignRoles( VecHeroes & heroes );

    // secondary_min=1 and `hero` is not the champion.
    bool keepsMinimalArmy( const Heroes & hero );

    // Moves the troops of a hero's army to `receiver` (a hero's army or a castle garrison), keeping one
    // monster of the fastest kind among the weaker half of the hero's monsters.
    void handOverArmy( Army & giver, Army & receiver );

    // split_singles=2: the fastest stack of the weaker half of the monster kinds, except the strongest stack (ties: the
    // weaker monster), is split into single
    // monsters in the free slots, at least one monster stays in the stack.
    void splitFastStackIntoFreeSlots( Army & army );

    // split_singles=1 for the battle of `army` (Battle::Loader): splits on construction, merges back on destruction.
    class BattleSplit
    {
    public:
        explicit BattleSplit( Army & army );
        BattleSplit( const BattleSplit & ) = delete;
        BattleSplit & operator=( const BattleSplit & ) = delete;
        ~BattleSplit();

    private:
        Army & _army;
        bool _split = false;
    };
}
