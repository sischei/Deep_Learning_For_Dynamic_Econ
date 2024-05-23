import importlib
import tensorflow as tf
import Parameters
from Graphs import do_random_step
import sys

"""
Example file for impulse response 
"""

import Globals
Globals.POST_PROCESSING=True

# make sure we can change function run profiles (e.g. shocks or no shocks)
tf.config.experimental_run_functions_eagerly(True)
tf.get_logger().setLevel('CRITICAL')

Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
starting_policy = Parameters.policy(Parameters.starting_state)

## simulate from a single starting state which is the mean across the N_sim_batch individual trajectory starting states
simulation_starting_state = tf.math.reduce_mean(Parameters.starting_state, axis = 0, keepdims=True)
## simulate a long range and calculate variable bounds + means from it for plotting
print("Running a one step with shocks turned on")       
shocked_state = do_random_step(simulation_starting_state)

## to just perturb one single state, i.e, apply a single shock: do State.update in the impulse_reponse script, so you can manipulate the "fully" shocked state as you wish (e.g. take only a coordinate from it before doing the non-shocked rounds - and taking the others from the original starting point, or you can take multiple ones as well, etc...)


# turn off shocks
Globals.DISABLE_SCHOCKS = True

# now do impulse response
for i in range(20):
    shocked_state = do_random_step(shocked_state)
    print(shocked_state, i)

del sys.modules['Parameters']
