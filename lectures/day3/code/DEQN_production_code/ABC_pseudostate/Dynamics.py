import tensorflow as tf
import math
import itertools
import Definitions
import State
import Parameters
from Parameters import sigma_ax, sigma_dx, rho_ax, rho_dx, alpha, alpha_low, alpha_up, beta, beta_low, beta_up, delta, tau, A_star
import PolicyState
from scipy.stats import norm
import Globals

# shocks
shocks_ax = [x * math.sqrt(2.0) * sigma_ax for x in [-1.224744871, 0.0, +1.224744871]]
probs_ax = [x / math.sqrt(math.pi) for x in [0.2954089751, 1.181635900, 0.2954089751]]

shocks_dx = [x * math.sqrt(2.0) * sigma_dx for x in [-1.224744871, 0.0, +1.224744871]]
probs_dx = probs_ax


shock_values = tf.constant(list(itertools.product(shocks_ax, shocks_dx)))
shock_probs = tf.constant([ p_ax * p_dx for p_ax, p_dx in list(itertools.product(probs_ax, probs_dx))])

if Parameters.expectation_type == 'monomial':
   shock_values, shock_probs = State.monomial_rule([sigma_ax, sigma_dx])

# helper functions 
N_executions = tf.Variable(0, dtype=tf.int32)
T = tf.Variable(Parameters.N_episode_length -1, dtype=tf.int32)
def helper_count_N_executions(N_executions):
    # count the number of executions
    return N_executions.assign_add(1)

def helper_reset_N_executions(N_executions):
    # reset the number of executions
    return N_executions.assign(0)

def total_step_random(prev_state, policy_state):
    ar = AR_step(prev_state)
    
    # original version:
    # if not Globals.DISABLE_SCHOCKS:
    #     shock = shock_step_random(prev_state)
    
    # increment counter
    helper_count_N_executions(N_executions)
    #tf.print("current episode: ",current_state)
    if not Globals.DISABLE_SCHOCKS and N_executions % T ==0:
        shock = shock_step_random(prev_state)
        # update dummy_x to for exclusion in Loss function
        #shock = State.update(shock, "dummy_x", tf.ones_like(State.dummy_x(prev_state)))
        

        # reset counter
        helper_reset_N_executions(N_executions)

    elif not Globals.DISABLE_SCHOCKS and N_executions % T !=0:
        shock = shock_step_random_exp_3(prev_state)
        shock = State.update(shock, "dummy_x", tf.ones_like(State.dummy_x(prev_state)))
    
    elif Globals.DISABLE_SCHOCKS == True and Globals.PROD_SHOCK == True and Globals.PREF_SHOCK == False: 
 
        # experiment 1
        shock = shock_step_random_exp_1(prev_state) #shock productivity
        print("one timestep with IRS -- a_x shock")

    elif Globals.DISABLE_SCHOCKS == True and Globals.PROD_SHOCK == False and Globals.PREF_SHOCK == True: 
 
        # experiment 2
        shock = shock_step_random_exp_2(prev_state) #shock to preferences 
        print("one timestep with IRS -- d_x shock")
        
    elif Globals.DISABLE_SCHOCKS  == True and Globals.PROD_SHOCK == False and Globals.PREF_SHOCK == False and Globals.PSEUDO_SHOCK == False:
        shock = tf.zeros_like(prev_state)   
        shock = State.update(shock, "alpha_x", State.alpha_x(prev_state))    
        shock = State.update(shock, "beta_x" , State.beta_x(prev_state))
        #tf.print(shock,"switched off") 

    elif Globals.DISABLE_SCHOCKS  == True and Globals.PROD_SHOCK == True and Globals.PREF_SHOCK == True and Globals.PSEUDO_SHOCK == False:
        shock = shock_step_random_exp_3(prev_state)
        #tf.print(shock,"switched off") 

    elif Globals.DISABLE_SCHOCKS  == True and Globals.PROD_SHOCK == True and Globals.PREF_SHOCK == True and Globals.PSEUDO_SHOCK == True:
        shock = shock_step_random(prev_state)
        shock = State.update(shock, "dummy_x", tf.ones_like(State.dummy_x(prev_state)))
        #tf.print("all states are shocked (including pseudo state)")
    else:
        shock = shock_step_random(prev_state)

    policy = policy_step(prev_state, policy_state)
    
    total = ar + shock + policy  

    next_state = augment_state(total) 
    return next_state

# same as above, but non randomized shock, but rather the same shock for each realization
def total_step_spec_shock(prev_state, policy_state, shock_index):
    ar = AR_step(prev_state)
    shock = shock_step_spec_shock(prev_state, shock_index)
    policy = policy_step(prev_state, policy_state)
    
    total = ar + shock + policy        
    return augment_state(total)

def augment_state(state):
    state = State.update(state, "d_x", tf.math.exp(State.d_x(state)))
    return State.update(state, "a_x", tf.math.exp(State.a_x(state)))

def AR_step(prev_state):
    # only rf, yt and kappa have autoregressive components
    ar_step = tf.zeros_like(prev_state)
    ar_step = State.update(ar_step, "a_x", Parameters.rho_ax * tf.math.log(State.a_x(prev_state)) + (1 -  Parameters.rho_ax) * tf.math.log(Parameters.A_star))
    ar_step = State.update(ar_step, "d_x", Parameters.rho_dx * tf.math.log(State.d_x(prev_state)) )
    return ar_step


def shock_step_random(prev_state):
    shock_step = tf.zeros_like(prev_state)
    random_normals = Parameters.rng.normal([prev_state.shape[0],2])
    shock_step = State.update(shock_step, "a_x", random_normals[:,0] * Parameters.sigma_ax)
    shock_step = State.update(shock_step, "d_x", random_normals[:,1] * Parameters.sigma_dx)

    random_uniform = Parameters.rng.uniform([prev_state.shape[0],2])
    shock_step = State.update(shock_step, "alpha_x", random_uniform[:,0]*(alpha_up-alpha_low) + alpha_low)    
    shock_step = State.update(shock_step, "beta_x" , random_uniform[:,1]*(beta_up-beta_low) + beta_low)

    return shock_step


#================================================================== 
#   Special section in Dynamics, concerned with impulse-response 
#   experiment 1 - a_x_state
#   experiment 2 - d_x_state
#   HARD-coded -- please add above manually the function call 
 
#------------------------------------------------------------------        
def shock_step_random_exp_1(prev_state):
    """
    shock to a_x_state
    
    """
    
    shock_step = tf.zeros_like(prev_state)

    # quantile -- 1 std shock
    # 1 std: 0.84134; 2 std: 0.97725 3 std: 0.99865
    quantile = 0.84134
    IRS_shock = norm.ppf([quantile]) #inverse quantile
    #print(norm.cdf(norm.ppf(quantile)))


    i = 1 # shock in country i (USA:1,...)
    vola_ax = 0.0
    if(i == 1):
        vola_ax = sigma_ax
        print("ax shock -- experiment 1")

    #print(sigma_ax)
    shock_step = State.update(shock_step,'a_x',sigma_ax*IRS_shock)
    #print(shock_step)
    shock_step = State.update(shock_step, "alpha_x", State.alpha_x(prev_state))    
    shock_step = State.update(shock_step, "beta_x" , State.beta_x(prev_state))
    
    return shock_step


def shock_step_random_exp_2(prev_state):
    """
    shock to d_x_state
    
    """
    
    shock_step = tf.zeros_like(prev_state)

    # quantile -- 1 std shock
    # 1 std: 0.84134; 2 std: 0.97725 3 std: 0.99865
    quantile = 0.84134
    IRS_shock = norm.ppf([quantile]) #inverse quantile
    #print(norm.cdf(norm.ppf(quantile)))


    i = 1 # shock in country i (USA:1,...)
    vola_dx = 0.0
    if(i == 1):
        vola_dx = sigma_dx
        print("ax shock -- experiment 1")

    #print(sigma_ax)
    shock_step = State.update(shock_step,'d_x',sigma_dx*IRS_shock)
    #print(shock_step)
    shock_step = State.update(shock_step, "alpha_x", State.alpha_x(prev_state))    
    shock_step = State.update(shock_step, "beta_x" , State.beta_x(prev_state))
    
    return shock_step

def shock_step_random_exp_3(prev_state):
    """
    shock to a_x_state and d_x_state
    
    """ 
    
    shock_step = tf.zeros_like(prev_state)
    
    random_normals = Parameters.rng.normal([prev_state.shape[0],2])
    shock_step = State.update(shock_step, "a_x", random_normals[:,0] * Parameters.sigma_ax)
    shock_step = State.update(shock_step, "d_x", random_normals[:,1] * Parameters.sigma_dx)

    shock_step = State.update(shock_step, "alpha_x", State.alpha_x(prev_state))    
    shock_step = State.update(shock_step, "beta_x" , State.beta_x(prev_state))
    

    
    return shock_step

def shock_step_spec_shock(prev_state, shock_index):
    # Use a specific shock - for calculating expectations
    shock_step = tf.zeros_like(prev_state)
    shock_step = State.update(shock_step,"a_x", tf.repeat(shock_values[shock_index,0], prev_state.shape[0]))
    shock_step = State.update(shock_step,"d_x", tf.repeat(shock_values[shock_index,1], prev_state.shape[0]))

    shock_step = State.update(shock_step, "alpha_x", State.alpha_x(prev_state))
    shock_step = State.update(shock_step, "beta_x" , State.beta_x(prev_state))
    return shock_step

def policy_step(prev_state, policy_state):
    # coming from the lagged policy / definition
    policy_step = tf.zeros_like(prev_state)
    policy_step = State.update(policy_step, "K_x",PolicyState.Ishare_y(policy_state) * (Definitions.Y_x(prev_state,policy_state)) + (1 - Parameters.delta) * State.K_x(prev_state))
    
 
    return policy_step
    
