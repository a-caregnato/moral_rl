from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import numpy as np
import pickle
import copy
import argparse
import sys

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import rendering
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
This is a modification of the environment proposed in Peshcl et. al (2022) 
for a multi-target problem in a small grid (5x5) with one target TYPE1, 
one target TYPE2, and a few obstacles


"""

# General Parameters
MAX_STEPS = 75
N_TAR_A = 3
N_TAR_B = 2
GRID_X_MAX = 18
GRID_Y_MAX = 10

"""
WAREHOUSE_ART = \
    ['####################',
     '#   O     O        #',
     '#   O     O        #',
     '#   O     O        #',
     '#   O     O        #',
     '#                  #',
     '#               O  #',
     '#               OOO#',
     '#               O  #',
     '#O  O        O     #',
     '#   O        O     #',
     '#   O        O     #',
     '#   O        O     #',
     '####################']

BACKGROUND_ART = \
    ['####################',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '####################']
"""

WAREHOUSE_ART = \
    ['####################',
     '#  O               #',
     '#  O               #',
     '#                  #',
     '#                  #',
     '#OO                #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '####################']

BACKGROUND_ART = \
    ['####################',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '#                  #',
     '####################']


WAREHOUSE_FG_COLOURS = {' ': (999, 999, 999),  # Floor.
                        '#': (428, 135, 0),    # Walls.
                        'x': (850, 603, 270),  # Unused.
                        'P': (388, 400, 999),  # The player.
                        'O': (0, 0, 0),  # Obstacle
                        'A': (0, 999, 0), # Target type A
                        'B': (999, 0, 0) # Target type B
                        }    


def make_game(seed=None, demo=False):
    warehouse_art = WAREHOUSE_ART
    what_lies_beneath = BACKGROUND_ART
    sprites = {'P': PlayerSprite}

    if demo:
        raise NotImplementedError
    else:
        drapes = {'X': JudgeDrape}

    drapes['O'] = ObstacleDrape
    drapes['A'] = TargetADrape
    drapes['B'] = TargetBDrape

    update_schedule = [['O'],
                       ['A'],
                       ['B'],
                       ['X'],
                       ['P']]


    return ascii_art.ascii_art_to_game(
        warehouse_art, what_lies_beneath, sprites, drapes,
        update_schedule=update_schedule)


def scalar_to_idx(x):
    row = x%GRID_Y_MAX
    col = int(np.floor(x/GRID_X_MAX))
    return (row+1, col+1)


class ObstacleDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(ObstacleDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.
    
        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:
            the_plot.add_reward(np.array([0., -1, 0., 0.]))


class TargetADrape(plab_things.Drape): #(green target)
    def __init__(self, curtain, character):
        super(TargetADrape, self).__init__(curtain, character)
        self.curtain.fill(False) # randomizes position afterwards

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop

        if the_plot.frame == 0:
            # Random initialization of player, fire and citizen
            n_assigned_targets = 0
            
            while n_assigned_targets <= N_TAR_A-1:
                random_positions = np.random.randint(0,GRID_X_MAX*GRID_Y_MAX, size=1)[0]
                tmp_idx = scalar_to_idx(random_positions)
                not_obstacle = not layers['O'][tmp_idx] # checks if chosen target coordinate is in an obtacle
                out_of_bounds = layers['#'][tmp_idx]
                not_target = not self.curtain[tmp_idx]                
                if not_target and not_obstacle and not out_of_bounds:
                    self.curtain[tmp_idx] = True
                    n_assigned_targets = n_assigned_targets + 1
            
            #self.curtain[(4,3)] = True
        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        # Checks if at target type A
        if self.curtain[(player_row, player_col)]:
             self.curtain[(player_row, player_col)] = False
             the_plot.add_reward(np.array([0., 0., 0.5, 0.]))


class TargetBDrape(plab_things.Drape):  #(red target)
    def __init__(self, curtain, character):
        super(TargetBDrape, self).__init__(curtain, character)
        self.curtain.fill(False) # randomizes position afterwards

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop
        if the_plot.frame == 0:
            n_assigned_targets = 0
            
            
            while n_assigned_targets <= N_TAR_B-1:
                random_positions = np.random.randint(0,GRID_X_MAX*GRID_Y_MAX, size=1)[0]
                tmp_idx = scalar_to_idx(random_positions)
                not_obstacle = not layers['O'][tmp_idx] # checks if chosen target coordinate is in an obtacle
                out_of_bounds = layers['#'][tmp_idx]
                not_target = not layers['A'][tmp_idx] and not self.curtain[tmp_idx]
                if not_target and not_obstacle and not out_of_bounds:
                    self.curtain[tmp_idx] = True
                    n_assigned_targets = n_assigned_targets + 1
            # last target of type B is terminal, saves its idx
            self.last_tar_coord = np.array(tmp_idx)
            
            #self.curtain[(1,5)] = True
            #self.last_tar_coord = np.array((1,5))

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

         # Termination condition over targets
        if self.curtain[(player_row, player_col)]:
            #self.curtain[(player_row, player_col)] = False # remove tar
            the_plot.add_reward(np.array([0., 0., 0., 0.25]))
            if np.array_equal(np.array([player_row, player_col]),self.last_tar_coord): # checks if terminal condition
                the_plot.terminate_episode()

class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#.')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things  # Unused.
        
        # weirdly rng ini pos of agent is set here
        if the_plot.frame == 0:
            ini_pos_assigned = False
            
            
            while not ini_pos_assigned:
                random_positions = np.random.randint(0,GRID_X_MAX*GRID_Y_MAX, size=1)[0]
                tmp_idx = scalar_to_idx(random_positions)
                not_obstacle = not layers['O'][tmp_idx]
                out_of_bounds = layers['#'][tmp_idx]
                not_target = not layers['B'][tmp_idx] and not layers['A'][tmp_idx]
                if not_target and not_obstacle and not out_of_bounds:
                    the_plot['P_pos'] = tmp_idx
                    ini_pos_assigned = True
                    self._teleport(the_plot['P_pos'])
            

            #the_plot['P_pos'] = (3,1)
            #self._teleport(the_plot['P_pos'])

        if actions == 0:    # go upward?
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            self._east(board, the_plot)
    
        if the_plot.frame > 0:
            the_plot.add_reward(np.array([-0.02, 0., 0., 0.]))


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS
        

    def update(self, actions, board, layers, backdrop, things, the_plot):

        #if the_plot.frame == 0:
            #self.last_pos = the_plot['P_pos']

        #player_pattern_position = things['P'].position
        #pos = np.array([player_pattern_position.row,player_pattern_position.col])

        #self.last_pos = 


        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        #if the_plot.frame > 0:
            #the_plot.add_reward(np.array([-0.01, 0., 0.]))
        #the_plot.add_reward(-0.1)
        self._step_counter += 1

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if self._step_counter == self._max_steps:
            the_plot.terminate_episode()


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)


    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         'w': 5,
                         's': 6,
                         'a': 7,
                         'd': 8,
                         -1: 4,
                         'q': 9, 'Q': 9},
        delay=1000,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo', help='Record demonstrations',
                        action='store_true')
    args = parser.parse_args()
    if args.demo:
        main(demo=True)
    else:
        main(demo=False)
