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

namespace Battle
{
    class Arena;
    class Command;
    class Unit;

    // Serializes the current battle state into the wire format shared by the headless battle
    // server and the real-battle agent protocol (units, obstacles, legal moves, result). The
    // state reply schema is documented in az/README.md.
    std::string SerializeArenaState( Arena & arena, const Unit * currentUnit, const std::vector<Command> & legalMoves );

    // Enumerates the moves available to the unit at the current decision point (MOVE to every
    // reachable cell, ATTACK from every reachable cell or as a shooter, SKIP). Shared by the
    // headless battle server and the real-battle agent protocol.
    std::vector<Command> EnumerateLegalMoves( Arena & arena, const Unit & unit );

    // Runs the headless battle server (JSON lines on stdin/stdout) used by the AlphaZero-style
    // battle prototype (see az/README.md). Enabled by the FHEROES2_BATTLE_SERVER environment
    // variable; returns false immediately when it is not set, otherwise never returns until
    // the client sends the "quit" operation.
    bool RunBattleServer();
}
