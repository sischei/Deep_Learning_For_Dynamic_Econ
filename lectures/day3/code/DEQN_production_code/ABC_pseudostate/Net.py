import tensorflow as tf
import importlib
import keras

def define_net(cfg,states,policy_states):
    MODEL_NAME_ = cfg.MODEL_NAME
    variables = importlib.import_module(MODEL_NAME_ + ".Variables")
    state_space_dim_ = len(variables.states)

    tf.keras.backend.set_floatx(cfg.run.get('keras_precision','float32'))

    ### MOD FOR CUSTOM NET ###
    # get lists from Variables.py in the model's directory
    # list of states; those lists can overlap and are subset of list states and union is equal to states
    config_states_econ = variables.states_x 
    config_states_dummy = variables.dummy_x_state

    # list of  policy states; those lists are disjoint and union is equal to policy_state
    config_policies_econ = variables.econ_policies
    config_policies_dummy = variables.dummy_y_policy

    #get the list of indices; i.e. what place do the agent/econ states have in states list
    states_econ_ = [s['name'] for s in config_states_econ]
    states_dummy_ = [s['name'] for s in config_states_dummy]
    state_indx_dummy = []
    for indxec in states_dummy_:
        state_indx_dummy.append(states.index(indxec))

    state_indx_econ = []
    for indxag in states_econ_:
        state_indx_econ.append(states.index(indxag))

    #get the list of indices; i.e. what place do the agent/econ policy_states have in policy_states list
    policy_states_econ_ = [s['name'] for s in config_policies_econ]
    policy_states_dummy_ = [s['name'] for s in config_policies_dummy]
    policy_indx_dummy = []
    for indxec in policy_states_dummy_:
        policy_indx_dummy.append(policy_states.index(indxec))

    policy_indx_econ = []
    for indxag in policy_states_econ_:
        policy_indx_econ.append(policy_states.index(indxag))

    x_in = keras.Input(shape=(state_space_dim_,)) #define input vector variable with same dimension as state_space_dim

    #Define agents' net layer by layer
    x_econ = tf.gather(x_in,state_indx_econ,axis=-1) #select subset of states to send to agent net

    hidden_layer_econ_out = tf.keras.layers.Dense(
        units = 256, 
        activation = 'selu', 
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.01, 
                                                                   mode='fan_in', 
                                                                   distribution='truncated_normal', seed=1))(x_econ)
    hidden_layer_econ_out = tf.keras.layers.Dense(
        units = 256, 
        activation = 'selu', 
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.01, 
                                                                   mode='fan_in', 
                                                                   distribution='truncated_normal', seed=2))(hidden_layer_econ_out)
    
    #output layer
    out_econ = tf.keras.layers.Dense(
        units = len(policy_states_econ_), 
        activation = 'linear', 
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.01, 
                                                                   mode='fan_in', 
                                                                   distribution='truncated_normal', seed=5))(hidden_layer_econ_out)
    
    #Define economie's net layer by layer
    x_dummy = tf.gather(x_in,state_indx_dummy,axis=-1) #select subset of states to send to economy net

    # hidden_layer_dummy_out = tf.keras.layers.Dense(
    #     units = 8, 
    #     activation = 'selu', 
    #     kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.01, 
    #                                                                mode='fan_in', 
    #                                                                distribution='truncated_normal', seed=6))(x_dummy)

    #output layer
    out_dummy = tf.keras.layers.Dense(
        units = len(policy_states_dummy_), 
        activation = 'linear', 
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.01, 
                                                                   mode='fan_in', 
    
                                                                   distribution='truncated_normal', seed=10))(x_dummy)
    
    #joint the final outputs together so they have the same dimension as policy_states vector
    output = tf.concat([out_econ,out_dummy],axis=-1)

    #define the model
    policy_net = keras.Model(inputs=x_in, outputs=output, name="mult_model")


    return policy_net