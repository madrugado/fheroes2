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

#include <string>
#include <vector>

class Army;
class Castle;

namespace Battle
{
    class Arena;
    class Command;
    class Unit;

    // Serializes the current battle state into the wire format shared by the headless battle
    // server and the real-battle agent protocol (units, obstacles, legal moves, result). The
    // state reply schema is documented in rl/README.md.
    std::string SerializeArenaState( Arena & arena, const Unit * currentUnit, const std::vector<Command> & legalMoves );

    // Enumerates the moves available to the unit at the current decision point (MOVE to every
    // reachable cell, ATTACK from every reachable cell or as a shooter, SKIP). Shared by the
    // headless battle server and the real-battle agent protocol.
    std::vector<Command> EnumerateLegalMoves( Arena & arena, const Unit & unit );

    // The hero spells the side to move may cast now (part of EnumerateLegalMoves()): one
    // SPELLCAST command per spell and target, filtered by the same rules as the spell book of the
    // battle interface (combat spell, castable, not disabled, a valid target).
    std::vector<Command> EnumerateSpellCasts( const Arena & arena );

    // The commander's escapes the side to move may take now (part of EnumerateLegalMoves(), after the
    // spells): RETREAT and SURRENDER with exactly the preconditions of Arena::ApplyActionRetreat() /
    // ApplyActionSurrender() (a hero commander; retreat not from a defended castle; surrender only
    // to a hero or captain and only when the kingdom can pay).
    std::vector<Command> EnumerateEscapes( const Arena & arena );

    // Hex-encoded save-game serialization of the army's commander hero (empty when the army has
    // no hero, e.g. neutral monsters or a castle garrison). The real-battle agent protocol sends
    // it in "battle_start"; the battle server "new" operation restores the hero from it, so the
    // agent's replica fights with the same primary/secondary skills, artifacts, spells and
    // morale/luck sources (visited objects) as the real battle.
    std::string EncodeCommander( const Army & army );

    // Hex-encoded save-game serialization of a castle or town (buildings, captain, garrison,
    // owner): sent in "battle_start" for battles on a castle/town tile and restored by the battle
    // server "new" operation (sieges, town garrisons with a captain, castle morale/luck).
    std::string EncodeCastle( const Castle & castle );

    // Runs the headless battle server (JSON lines on stdin/stdout) used by the AlphaZero-style
    // battle prototype (see rl/README.md). Enabled by the FHEROES2_BATTLE_SERVER environment
    // variable; returns false immediately when it is not set, otherwise never returns until
    // the client sends the "quit" operation.
    bool RunBattleServer();
}
