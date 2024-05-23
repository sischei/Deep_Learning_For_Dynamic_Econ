# -*- coding: utf-8 -*-
import tensorflow as tf
import math
import pandas as pd
import os

print("working directory",os.getcwd())
folder_nr = int(os.environ["FOLDER"])
param_df = pd.read_csv("../../../../comparison_ABC/alpha_beta_random.csv",usecols=range(1,3)).to_numpy()
N_param_combinations = param_df.shape[0]

################################################
# constants from config file

constants = {'alpha': param_df[folder_nr -1,0], #0.33,
            'beta': param_df[folder_nr -1,1], # 0.96,
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


###############################

### total state space
states = end_state + ex_state 

print("number of states", len(states))


#################################################
### Policies

Ishare_y_policy = []
Ishare_y_policy.append({'name':'Ishare_y','activation': 'tf.nn.sigmoid'})

Lamda_y_policy = []
Lamda_y_policy.append({'name':'Lamda_y','activation': 'tf.nn.softplus'})

### total number of policies
policies = Ishare_y_policy + Lamda_y_policy

print("number of policies", len(policies))


##################################################
### definitions

#pT_policy = []
#pT_policy.append({'name':'pT','activation': 'tf.nn.softplus'})
Y_x = [{'name':'Y_x','activation': 'tf.nn.softplus'}]
IC_y = [{'name':'IC_y'}]
### total number of definitions
definitions = Y_x + IC_y



print("number of definitions", len(definitions)) 
