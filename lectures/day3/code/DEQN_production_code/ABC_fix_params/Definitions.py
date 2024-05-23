import Parameters
import PolicyState
import State
import tensorflow as tf
from Parameters import sigma_ax, sigma_dx, rho_ax, rho_dx, alpha, beta, delta, tau, A_star


#def pT(state, policy_state):
    #derived quantities
#    return PolicyState.q_real(policy_state) * Parameters.pTstar  

# production in current period
def Y_x(state, policy_state=None):
    return State.a_x(state) * State.K_x(state) ** alpha
    
# binding constraint indicator
def IC_y(state, policy_state):
    return tf.cast(tf.math.greater(PolicyState.Lamda_y(policy_state), 0.001 + 0.003746), dtype=tf.int32)