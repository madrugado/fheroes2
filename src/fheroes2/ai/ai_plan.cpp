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

#include "ai_plan.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "army.h"
#include "army_troop.h"
#include "color.h"
#include "heroes.h"
#include "monster.h"

namespace
{
    struct Plan
    {
        std::string color; // empty = every player
        std::map<std::string, int> values;
    };

    const Plan & plan()
    {
        static const Plan parsed = [] {
            Plan result;
            const char * env = std::getenv( "FHEROES2_PLAN" );
            if ( env == nullptr ) {
                return result;
            }

            const std::string text( env );
            size_t start = 0;
            while ( start <= text.size() ) {
                const size_t end = std::min( text.find( ',', start ), text.size() );
                const std::string item = text.substr( start, end - start );
                const size_t equal = item.find( '=' );
                if ( equal != std::string::npos ) {
                    const std::string key = item.substr( 0, equal );
                    const std::string value = item.substr( equal + 1 );
                    if ( key == "color" ) {
                        result.color = value;
                    }
                    else {
                        result.values[key] = std::atoi( value.c_str() );
                    }
                }
                start = end + 1;
            }
            return result;
        }();
        return parsed;
    }

    // The main hero of every kingdom (by color), kept while the hero lives in the kingdom.
    std::map<PlayerColor, int> & championIds()
    {
        static std::map<PlayerColor, int> ids;
        return ids;
    }
}

int AIPlan::value( const PlayerColor color, const char * key )
{
    const Plan & current = plan();
    if ( current.values.empty() || ( !current.color.empty() && current.color != Color::String( color ) ) ) {
        return 0;
    }

    const auto it = current.values.find( key );
    return it == current.values.end() ? 0 : it->second;
}

bool AIPlan::assignRoles( VecHeroes & heroes )
{
    if ( heroes.empty() || value( heroes.front()->GetColor(), "champion" ) == 0 ) {
        return false;
    }

    const PlayerColor color = heroes.front()->GetColor();
    int & championId = championIds()[color];

    Heroes * champion = nullptr;
    for ( Heroes * hero : heroes ) {
        if ( hero->GetID() == championId && !hero->Modes( Heroes::PATROL ) ) {
            champion = hero;
        }
    }

    if ( champion == nullptr ) {
        for ( Heroes * hero : heroes ) {
            if ( hero->Modes( Heroes::PATROL ) ) {
                continue;
            }
            if ( champion == nullptr || hero->GetArmy().GetStrength() > champion->GetArmy().GetStrength() ) {
                champion = hero;
            }
        }
    }

    for ( Heroes * hero : heroes ) {
        if ( hero->Modes( Heroes::PATROL ) ) {
            // Patrolling heroes can only fight (as in the built-in assignment).
            hero->setAIRole( Heroes::Role::FIGHTER );
        }
        else if ( hero == champion ) {
            hero->setAIRole( Heroes::Role::CHAMPION );
        }
        else {
            hero->setAIRole( Heroes::Role::COURIER );
        }
    }

    championId = champion != nullptr ? champion->GetID() : -1;
    return true;
}

bool AIPlan::keepsMinimalArmy( const Heroes & hero )
{
    if ( value( hero.GetColor(), "secondary_min" ) == 0 ) {
        return false;
    }

    // Only once the plan has chosen the champion: before the first role assignment of a game (e.g. the
    // castle turn of day 1) every hero still has the default role.
    const auto it = championIds().find( hero.GetColor() );
    return it != championIds().end() && it->second >= 0 && hero.GetID() != it->second;
}

void AIPlan::handOverArmy( Army & giver, Army & receiver )
{
    std::vector<Troop *> troops;
    uint32_t monsters = 0;
    for ( size_t i = 0; i < giver.Size(); ++i ) {
        Troop * troop = giver.GetTroop( i );
        if ( troop != nullptr && troop->isValid() ) {
            troops.push_back( troop );
            monsters += troop->GetCount();
        }
    }

    if ( monsters <= 1 ) {
        return; // nothing to hand over
    }

    // One monster of the fastest kind among the weaker half (by the strength of one monster) stays.
    std::vector<double> strengths;
    strengths.reserve( troops.size() );
    for ( const Troop * troop : troops ) {
        strengths.push_back( troop->GetMonsterStrength() );
    }
    std::vector<double> sorted = strengths;
    std::sort( sorted.begin(), sorted.end() );
    const double median = sorted[( sorted.size() - 1 ) / 2];

    Troop * kept = nullptr;
    double keptStrength = 0;
    for ( size_t i = 0; i < troops.size(); ++i ) {
        if ( strengths[i] > median ) {
            continue;
        }
        if ( kept == nullptr || troops[i]->GetSpeed() > kept->GetSpeed() || ( troops[i]->GetSpeed() == kept->GetSpeed() && strengths[i] < keptStrength ) ) {
            kept = troops[i];
            keptStrength = strengths[i];
        }
    }

    const Monster keptMonster( kept->GetID() );
    if ( kept->GetCount() > 1 ) {
        kept->SetCount( kept->GetCount() - 1 );
    }
    else {
        kept->Reset();
    }

    if ( giver.isValid() ) {
        receiver.JoinStrongestFromArmy( giver );
    }

    // The kept monster goes back; whatever the receiver could not take stays with the giver.
    giver.JoinTroop( keptMonster, 1, false );

    for ( size_t i = 0; i < giver.Size(); ++i ) {
        Troop * troop = giver.GetTroop( i );
        if ( troop == nullptr || !troop->isValid() || troop->GetID() == keptMonster.GetID() || !receiver.CanJoinTroop( *troop ) ) {
            continue;
        }
        if ( receiver.JoinTroop( *troop ) ) {
            troop->Reset();
        }
    }
}
