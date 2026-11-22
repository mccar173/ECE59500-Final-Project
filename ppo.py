import argparse
import pprint

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed

uno_env = rlcard.make('uno')


print("Number of actions:", uno_env.num_actions)
print("Number of players:", uno_env.num_players)
print("Shape of state:", uno_env.state_shape)
print("Shape of action:", uno_env.action_shape)
     