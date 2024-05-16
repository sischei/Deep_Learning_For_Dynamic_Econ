"""
Filename: post_process_generic.py
Description:
Post processing with the trained (optimal solution) policy fuctions using DEQN.
With this script, we
- compute the dynamics of the exogenous parameters
- compute the distributions of the state, policy and (selected)
defined variables
- compute Euler errors of the loss equations
"""

import numpy as np  # using float32 to have a compatibility with tensorflow
import pandas as pd
import shutil
import importlib
import pandas as pd
import tensorflow as tf
import Parameters
import matplotlib.pyplot as plt
from matplotlib import rc
import State
import PolicyState
import Definitions
from Graphs import run_episode

tf.get_logger().setLevel('CRITICAL')
pd.set_option('display.max_columns', None)

# --------------------------------------------------------------------------- #
# Plot setting
# --------------------------------------------------------------------------- #
# Get the size of the current terminal
terminal_size_col = shutil.get_terminal_size().columns

exparams = ['tfp', 'gr_tfp', 'lab', 'gr_lab', 'sigma', 'theta1', 'Eland',
            'Fex', 'beta_hat']
econ_defs = ['con', 'Omega', 'Theta', 'ygross', 'ynet', 'inv', 'Eind',
            'scc', 'carbontax', 'Abatement', 'Dam', 'Emissions']

# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Simulate the dynamics of the exogenous parameters")
# --------------------------------------------------------------------------- #
alpha = Parameters.alpha
starting_state = Parameters.starting_state
starting_policy = Parameters.policy(starting_state)

Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

# Extract parameters
psi = Parameters.psi

N_state = len(Parameters.states)  # Number of state variables
N_policy_state = len(Parameters.policy_states)  # Number of policy variables
N_defined = len(Parameters.definitions)  # Number of defined variables

# Simulate the economy for N_simulated episode length
N_episode_length = Parameters.N_episode_length
starting_state = tf.reshape(tf.constant([
    Parameters.k0, Parameters.MAT0, Parameters.MUO0, Parameters.MLO0,
    Parameters.TAT0, Parameters.TOC0,
    Parameters.tau0]), shape=(1, N_state))

# Simulate the economy for N_episode_length time periods
simulation_starting_state = tf.tile(tf.expand_dims(
    starting_state, axis=0), [N_episode_length, 1, 1])
state_1episode = tf.reshape(
    run_episode(simulation_starting_state), [N_episode_length, N_state])
policy_1episode = Parameters.policy(state_1episode)

# Time periods
ts = Definitions.tau2t(state_1episode, policy_1episode)
df_time = pd.DataFrame()
df_time['time'] = ts
df_time.to_csv(Parameters.LOG_DIR + "/time.csv", index=False)

df_exopar = pd.DataFrame()

for de in exparams:
    de_val = getattr(Definitions, de)(state_1episode, policy_1episode)
    if de in ['gr_tfp', 'lab', 'gr_lab', 'theta1', 'Fex', 'beta_hat']:
        de_val = de_val
    elif de in ['tfp']:
        de_val =  (1000 * de_val)**(1-alpha)
    elif de in ['sigma', 'Eland']:
        de_val = 1000 * 3.666 * de_val

    df_exopar[de] = de_val

df_exopar.to_csv(Parameters.LOG_DIR + "/exoparams.csv", index=False)

# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Simulate one path. If there is no stochastic shock, it is "
      "equivalent to the deterministic path")
# --------------------------------------------------------------------------- #
df_states = pd.DataFrame()
for sidx, state in enumerate(Parameters.states):
    # State variable
    state_val = getattr(State, state)(state_1episode)
    # Adjust state variables
    if state in ['kx']:
        # Multiply tfp * lab
        tfp = Definitions.tfp(state_1episode, policy_1episode)
        lab = Definitions.lab(state_1episode, policy_1episode)
        state_val = state_val * tfp * lab
    elif state in ['MATx', 'MUOx', 'MLOx']:
        # Rescale to GtC
        state_val = state_val * 1000

    df_states[state] = state_val
df_states.to_csv(Parameters.LOG_DIR + "/states.csv", index=False)

df_ps = pd.DataFrame()
for pidx, ps in enumerate(Parameters.policy_states):
    # policy variable
    ps_val = getattr(PolicyState, ps)(policy_1episode)
    # Adjust state variables
    if ps in ['kplusy']:
        tfp = Definitions.tfp(state_1episode, policy_1episode)
        lab = Definitions.lab(state_1episode, policy_1episode)
        gr_tfp = Definitions.gr_tfp(state_1episode, policy_1episode)
        gr_lab = Definitions.gr_lab(state_1episode, policy_1episode)
        ps_val = ps_val * tf.math.exp(gr_tfp + gr_lab) * tfp * lab

    df_ps[ps] = ps_val
df_ps.to_csv(Parameters.LOG_DIR + "/ps.csv", index=False)

df_def = pd.DataFrame()
for didx, de in enumerate(
        ['con', 'Omega', 'Theta', 'ygross', 'ynet', 'inv', 'Eind',
        'scc', 'carbontax', 'Abatement', 'Dam', 'Emissions']):
    # defined economic variable
    de_val = getattr(Definitions, de)(state_1episode, policy_1episode)
    if de in ['con', 'ygross', 'ynet', 'inv', 'Dam', 'Abatement']:
        tfp = Definitions.tfp(state_1episode, policy_1episode)
        lab = Definitions.lab(state_1episode, policy_1episode)
        de_val = de_val * tfp * lab
    elif de in ['Eind']:
        de_val = de_val * tfp * lab * 1000 * 3.666
    elif de in ['Emissions']:
        de_val = de_val * 1000 * 3.666
    elif de in ['carbontax', 'scc','alter_scc']:
        de_val = de_val / 3.666

    df_def[de] = de_val
df_def.to_csv(Parameters.LOG_DIR + "/defs.csv", index=False)



# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Simulate the economy forward in batch")
# --------------------------------------------------------------------------- #
# Number of simulation batch, it should be arbitrary but big enough
N_sim_batch = 20

# Store the simulated episodes
state_episode_batch = np.empty(
    shape=[N_episode_length, N_sim_batch, N_state], dtype=np.float32)
policy_episode_batch = np.empty(
    shape=[N_episode_length, N_sim_batch, N_policy_state], dtype=np.float32)

# Simulate the economy for N_sim_batch times to compute the collection of
# state and policy episodes
for batchidx in range(N_sim_batch):
    # Start from the same initial state, but the simulated episodes are
    # different from each other due to the random shocks
    state_episode = tf.reshape(
        run_episode(simulation_starting_state), [N_episode_length, N_state])
    policy_episode = Parameters.policy(state_episode)
    # Store the current episode
    state_episode_batch[:, batchidx, :] = state_episode
    policy_episode_batch[:, batchidx, :] = policy_episode

# Some variables need to be rescaled for plotting
state_episode_scaled = np.empty_like(state_episode_batch, dtype=np.float32)
policy_episode_scaled = np.empty_like(policy_episode_batch, dtype=np.float32)
defined_episode_scaled = np.empty(
    shape=[N_episode_length, N_sim_batch, N_defined], dtype=np.float32)

# State variables
for sidx, state in enumerate(Parameters.states):
    # Adjust state variables
    for batchidx in range(N_sim_batch):
        state_episode = state_episode_batch[:, batchidx, :]
        policy_episode = policy_episode_batch[:, batchidx, :]
        state_val = getattr(State, state)(state_episode)
        if state in ['kx']:
            # Multiply tfp * lab
            tfp = Definitions.tfp(state_episode, policy_episode)
            lab = Definitions.lab(state_episode, policy_episode)
            state_val = state_val * tfp * lab
        elif state in ['MATx', 'MUOx', 'MLOx']:
            # Rescale to GtC
            state_val = 1000. * state_val
        state_episode_scaled[:, batchidx, sidx] = state_val

# Policy variables
for pidx, policy in enumerate(Parameters.policy_states):
    # Adjust policy variables
    for batchidx in range(N_sim_batch):
        state_episode = state_episode_batch[:, batchidx, :]
        policy_episode = policy_episode_batch[:, batchidx, :]
        policy_val = getattr(PolicyState, policy)(policy_episode)
        if policy in ['kplusy']:
            tfp = Definitions.tfp(state_episode, policy_episode)
            lab = Definitions.lab(state_episode, policy_episode)
            gr_tfp = Definitions.gr_tfp(state_episode, policy_episode)
            gr_lab = Definitions.gr_lab(state_episode, policy_episode)
            policy_val = policy_val * tf.math.exp(gr_tfp + gr_lab) * tfp * lab
        elif policy in ['cony']:
            tfp = Definitions.tfp(state_episode, policy_episode)
            lab = Definitions.lab(state_episode, policy_episode)
            policy_val = policy_val * tfp * lab
        policy_episode_scaled[:, batchidx, pidx] = policy_val

# Defined economic variables
for didx, de in enumerate(econ_defs):
    # Adjust defined variables
    for batchidx in range(N_sim_batch):
        state_episode = state_episode_batch[:, batchidx, :]
        policy_episode = policy_episode_batch[:, batchidx, :]
        defined_val = getattr(Definitions, de)(state_episode, policy_episode)
        if de in ['con', 'ygross', 'ynet', 'inv', 'Dam', 'Abatement']:
            tfp = Definitions.tfp(state_1episode, policy_1episode)
            lab = Definitions.lab(state_1episode, policy_1episode)
            de_val = de_val * tfp * lab
        elif de in ['Eind']:
            de_val = de_val * tfp * lab * 1000 * 3.666
        elif de in ['Emissions']:
            de_val = de_val * 1000 * 3.666
        elif de in ['carbontax', 'scc']:
            de_val = de_val / 3.666
        defined_episode_scaled[:, batchidx, didx] = defined_val


# We are interested in the Euler errors only from 2005 to 2100; therefore, we
# terminate this script if we simulate the economy not for 96 years.
ts_beg = 2015
ts_end = 2100
err_percentiles = [.001, 0.25, 0.50, 0.75, 0.999]
# ----------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Compute the Euler discrepancies until 2100")
# ----------------------------------------------------------------------- #
# Take the absolute numericl value of each element
for batchidx in range(N_sim_batch):
    state_episode = state_episode_batch[:, batchidx, :]
    policy_episode = policy_episode_batch[:, batchidx, :]
    euler_discrepancy_df = pd.DataFrame(
        Equations.equations(state_episode, policy_episode)).abs()
    state_episode_df = pd.DataFrame(
        {s: getattr(State, s)(state_episode) for s in Parameters.states})
    policy_episode_df = pd.DataFrame(
        {ps: getattr(PolicyState, ps)(policy_episode)
         for ps in Parameters.policy_states})
    defined_episode_df = pd.DataFrame(
        {de: getattr(Definitions, de)(state_episode, policy_episode)
         for de in econ_defs})

    # Initialize each dataframe
    if batchidx == 0:
        euler_discrepancies_df = euler_discrepancy_df
        state_episodes_df = state_episode_df
        policy_episodes_df = policy_episode_df
        defined_episodes_df = defined_episode_df
    else:
        euler_discrepancies_df = pd.concat([
            euler_discrepancies_df, euler_discrepancy_df], axis=0)
        state_episodes_df = pd.concat([
            state_episodes_df, state_episode_df], axis=0)
        policy_episodes_df = pd.concat([
            policy_episodes_df, policy_episode_df], axis=0)
        defined_episodes_df = pd.concat([
            defined_episodes_df, defined_episode_df], axis=0)

# Pring the Euler approximation errors
print("-" * terminal_size_col)
print("Print the Euler discrepancies")
print(euler_discrepancies_df.describe(percentiles=err_percentiles, include='all'))


# Save all relevant quantities along the trajectory
euler_discrepancies_df.to_csv(
    Parameters.LOG_DIR + "/simulated_euler_discrepancies_" + str(ts_beg) + '-' + str(ts_end) + ".csv", index=False, float_format='%.3e')

euler_discrepancies_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/simulated_euler_discrepancies_describe_" + str(ts_beg) + '-' + str(ts_end) + ".csv",
    index=True, float_format='%.3e')


print("-" * terminal_size_col)
print("Finished calculating Euler discrepancies and exit")
