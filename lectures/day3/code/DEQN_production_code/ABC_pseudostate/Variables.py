# -*- coding: utf-8 -*-
import tensorflow as tf
import math

################################################
# constants from config file

constants = {'alpha': 0.3, 
             # lower and upper bounds if alpha is a pseudo state
            'alpha_low':0.22, 
            'alpha_up':0.38, 
            'beta': 0.85,
            # lower and upper bounds if beta is a pseudo state
            'beta_low':0.75, 
            'beta_up':0.95,
            'delta': 0.1,
            'tau': 1.0,
            'A_star': 1.0,
            'sigma_ax': 0.04,
            'sigma_dx': 0.02,
            'rho_ax': 0.9,
            'rho_dx': 0.8}
    

#################################################
### states 

### endogenous states 
K_x_state = []
K_x_state.append({'name':'K_x','activation': 'tf.nn.softplus'})


# total endogenous state space
end_state = K_x_state


### exogenous states
a_x_state = []
a_x_state.append({'name':'a_x'})

d_x_state = []
d_x_state.append({'name':'d_x'})

# total endogenous state space
ex_state  = a_x_state + d_x_state

# total economic state space
econ_states = end_state + ex_state

### pseudo - states/ parameters as state variables
alpha_x_state = []
alpha_x_state.append({'name':'alpha_x', 'init':{'distribution':'uniform','kwargs': {'minval':0.22,'maxval':0.38}}})

beta_x_state = []
beta_x_state.append({'name':'beta_x', 'init':{'distribution':'uniform','kwargs': {'minval':0.75,'maxval':0.95}}})

# total pseudo state space
param_states  = alpha_x_state + beta_x_state

### dummy state
# Here we create a dummy state that is either 1 or 0. 0 in the first and last period in an episode, else 1. In the loss function it is then multiplied with the loss in order to set the loss in the first and last periods to 0.
# This ensures that the optimizer always has the same pseudostate in a given training sample.
# In post_processing, the dummy always needs to be set to 1 (see dynamics) in order to get the correct loss.
dummy_x_state = []
#dummy_x_state.append({'name':'dummy_x', 'init':{'distribution':'truncated_normal','kwargs': {'mean':1.0000,'stddev':0.00000}}})
dummy_x_state.append({'name':'dummy_x', 'init':{'distribution':'truncated_normal','kwargs': {'mean':0.0000,'stddev':0.00000}}})

###############################

states_x = econ_states + param_states

### total state space
states = states_x + dummy_x_state

print("number of states", len(states))


#################################################
### Policies

Ishare_y_policy = []
Ishare_y_policy.append({'name':'Ishare_y','activation': 'tf.nn.sigmoid'})

Lamda_y_policy = []
Lamda_y_policy.append({'name':'Lamda_y','activation': 'tf.nn.softplus'})

dummy_y_policy = []
dummy_y_policy.append({'name':'dummy_y'})

econ_policies = Ishare_y_policy + Lamda_y_policy

### total number of policies
policies = econ_policies + dummy_y_policy

print("number of policies", len(policies))


##################################################
### definitions

#pT_policy = []
#pT_policy.append({'name':'pT','activation': 'tf.nn.softplus'})
Y_x = [{'name':'Y_x','activation': 'tf.nn.softplus'}]

IC_y = [{'name':'IC_y'}]

CK_y = [{'name':'CK_y','activation': 'tf.nn.softplus'}]

#dummy_loss = [{'name':'dummy_loss'}]
### total number of definitions
definitions = Y_x + IC_y + CK_y #+ dummy_loss


print("number of definitions", len(definitions)) 
