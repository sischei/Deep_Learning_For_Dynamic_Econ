# ---------------------------------------------
# Purpose of file: API julia-DEQN. Allows to send states from julia to python and policies from DEQN solution are returned
# Required: DEQN solution, python main: post_process_jl.py
# Function input 'state': (key: state name, value: vector of state values) A dict that contains all states with the SAME KEY NAME as in the DEQN solution, the ordering of keys does not matter. 
# All states must be passed as keys in the dict. For each state, a vector must be passed (of at least length 1). Each state-variable vector must be of the same length.  
# Output: (key: policy name, value:vector of policy values) The function returns a Dict with the policy states. Each policy within the dict is returned as a vector.
# example of how to run this code (terminal):
#export USE_CONFIG_FROM_RUN_DIR=runs/ABC/testing && julia get_policy_state.jl STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR

using PyCall


# IS THIS NECESSARY?
pushfirst!(pyimport("sys")."path", "")

# THIS IS NECESSARY
dir = pwd()
pushfirst!(pyimport("sys")."path", dir)

# Load the Python module
py_post_process = pyimport("post_process_jl")

function call_get_policy_state_py(state)
    """
    Call the Python function Parameters.policy() with state as argument.

    Parameters:
    Dict{state}: Dict with vectors of state variables. ALL states must be passed to the function, each vector must be of the same length


    Returns:
    Dict{Array{Float32}}: The policies of the states as dict. Each policy within the dict is returned as a vector.
    """

    return py_post_process.get_policy_state(state)

end



# Example usage (with 6 states) EXAMPLES NEED TO BE UPDATED according to model at hand

# Example with 1 state coordinate
# define state
x_single = Dict(:K_x => [1.07749164], :a_x => [1.14276075], :d_x => [0.951961458], :alpha_x => [0.375841767], :beta_x => [0.757241726], :dummy_x => [1])

# call the function to get the policy
Policy_x_single = call_get_policy_state_py(x_single)

print("state: ",x_single)
print("Policy: ",Policy_x_single)

# Example with vectors of states
# define states (the state dict x can be combined from multiple dicts)
x_econ = Dict(:K_x => [1.07749164, 1.], :a_x => [1.14276075, 0.99], :d_x => [0.951961458, 1.01])
x_param = Dict(:alpha_x => [0.375841767, 0.375841767], :beta_x => [0.757241726, 0.757241726])
x_dummy = Dict(:dummy_x =>[1, 1]) # dummy is needed in ABC model, details see in ABC/Variables.py. To evaluate the loss function, the dummy needs to be set to 1

x = merge(x_econ, x_param, x_dummy)

# call the function to get the policy
Policy_x = call_get_policy_state_py(x)

print("state: ",x)
print("Policy: ",Policy_x)

