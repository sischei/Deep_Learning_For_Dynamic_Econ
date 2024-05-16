import tensorflow as tf
import numpy as np
import Parameters
import PolicyState
import State

#--------------------------------------------------------------------------- #
# Extract parameters
# --------------------------------------------------------------------------- #
# Dice specific parameter
Tstep = Parameters.Tstep

# Logarithmic time transformation
vartheta = Parameters.vartheta
# Economic parameters
rho = Parameters.rho
alpha, delta, psi = Parameters.alpha, Parameters.delta, Parameters.psi
# Population
L0, Linfty, deltaL = Parameters.L0, Parameters.Linfty, Parameters.deltaL
# TFP
A0hat, gA0hat, deltaA = Parameters.A0hat, Parameters.gA0hat, Parameters.deltaA
# Carbon intensity
sigma0, gSigma0, deltaSigma = Parameters.sigma0, Parameters.gSigma0, \
                                Parameters.deltaSigma
# Mitigation
theta2, pback, gback = Parameters.theta2, Parameters.pback, Parameters.gback
# Land emissions
ELand0, deltaLand = Parameters.ELand0, Parameters.deltaLand
# Exogenous forcings
fex0, fex1, Tyears = Parameters.fex0, Parameters.fex1, Parameters.Tyears
# Climate damage function
pi1, pi2, pow1, pow2 = Parameters.pi1, Parameters.pi2, Parameters.pow1, Parameters.pow2
# Carbon mass transitions
b12_, b23_, MATeq, MUOeq, MLOeq = Parameters.b12_, Parameters.b23_, \
    Parameters.MATeq, Parameters.MUOeq, Parameters.MLOeq
# Temperature exchange
c1_, c3_, c4_  = Parameters.c1_, Parameters.c3_, \
    Parameters.c4_
f2xco2, t2xco2 =  Parameters.f2xco2, Parameters.t2xco2
# Preindustrial carbon concentration
MATbase = Parameters.MATbase

# --------------------------------------------------------------------------- #
# Real and computational time periods
# --------------------------------------------------------------------------- #
def tau2t(state, policy_state):
    """ Scale back from the computational time tau to the real time t """
    _t = - tf.math.log(1 - State.taux(state)) / vartheta
    return _t


def tau2tauplus(state, policy_state):
    """ Update the computational time tau by tau + 1 based on the current real
    time t """
    _t = tau2t(state, policy_state)  # Current real time
    _tplus = _t + tf.ones_like(_t)  # Real time t + 1
    _tauplus = 1 - tf.math.exp(- vartheta * _tplus)  # Computational time tau+1
    return _tauplus

# --------------------------------------------------------------------------- #
# Exogenous parameters
# --------------------------------------------------------------------------- #
def tfp(state, policy_state):
    """ Deterministic TFP shock on the labor-argumented production function [-]
    """
    _t = tau2t(state, policy_state)
    _tfp = A0hat * tf.math.exp((Tstep * gA0hat) * (1 -
        tf.math.exp(-Tstep * deltaA * _t)) / (Tstep * deltaA))
    return _tfp


def gr_tfp(state, policy_state):
    """ Annual growth rate of the deterministic TFP shock [-/year] """
    _t = tau2t(state, policy_state)
    _gr_tfp = gA0hat * tf.math.exp(- Tstep *deltaA * _t)
    return _gr_tfp


def lab(state, policy_state):
    """ World population [million] """
    _t = tau2t(state, policy_state)
    _lab = L0 + (Linfty - L0) * (1 - tf.math.exp(-Tstep * deltaL * _t))
    return _lab


def gr_lab(state, policy_state):
    """ Annual growth rate of the world population [-/year] """
    _t = tau2t(state, policy_state)
    _gr_lab = (deltaL) / ((Linfty / (Linfty-L0)) * tf.math.exp(
                Tstep * deltaL * _t) - 1)
    return _gr_lab


def sigma(state, policy_state):
    """ Carbon intensity """
    _t = tau2t(state, policy_state)
    _sigma = sigma0 * tf.math.exp(Tstep * gSigma0 / np.log(1 + Tstep * deltaSigma)*
                    ((1 + Tstep * deltaSigma)**_t - 1))
    return _sigma


def theta1(state, policy_state):
    """ Cost coefficient of carbon mitigation """
    _t = tau2t(state, policy_state)
    _sigma = sigma(state, policy_state)
    c2co2 = Parameters.c2co2
    _theta1 = pback * (1000 * c2co2 *_sigma) * (tf.math.exp(-Tstep * gback * _t)) \
    / theta2
    return _theta1


def Eland(state, policy_state):
    """ Natural carbon emission """
    _t = tau2t(state, policy_state)
    _Eland = ELand0 * tf.math.exp(-Tstep * deltaLand * _t)
    return _Eland


def Fex(state, policy_state):
    """ External radiative forcing """
    _t = tau2t(state, policy_state)
    Year = np.int(Tyears / Tstep)
    _Fex = fex0 + (1 / Year) * (fex1 - fex0) * tf.math.minimum(_t, Year)
    return _Fex


def beta_hat(state, policy_state):
    """ Effective discout factor """
    _gr_tfp = gr_tfp(state, policy_state)
    _gr_lab = gr_lab(state, policy_state)
    _beta_hat = tf.math.exp((- rho  + (1-1/psi) * _gr_tfp + _gr_lab)* Tstep)
    return _beta_hat

def beta(state, policy_state):
    """ Discout factor """
    _gr_tfp = gr_tfp(state, policy_state)
    _gr_lab = gr_lab(state, policy_state)
    _beta = tf.math.exp((- rho )* Tstep)
    return _beta

def b12(state, policy_state):
    """ Mass of carbon transmission"""
    return Tstep * b12_

def b23(state, policy_state):
    """ Mass of carbon transmission"""
    return Tstep * b23_

def b21(state, policy_state):
    """ Mass of carbon transmission"""
    return MATeq/MUOeq * b12_* Tstep

def b32(state, policy_state):
    """ Mass of carbon transmission"""
    return MUOeq/MLOeq * b23_* Tstep

def c1c3(state, policy_state):
    return Tstep * c1_ * c3_

def c4(state, policy_state):
    return Tstep * c4_

def c1(state, policy_state):
    return Tstep * c1_

def c1f(state, policy_state):
    return Tstep * c1_ * f2xco2 / t2xco2

# --------------------------------------------------------------------------- #
# Economic variables
# --------------------------------------------------------------------------- #
def con(state, policy_state):
    """ Consumption policy """
    _lambd_haty = PolicyState.lambd_haty(policy_state)
    _con = _lambd_haty**(-psi)
    return _con

def Theta(state, policy_state):
    """ Abatement cost function """
    _theta1 = theta1(state, policy_state)
    _muy = PolicyState.muy(policy_state)
    _Theta = _theta1 * _muy**theta2
    return _Theta

def Theta_prime(state, policy_state):
    """ The first derivative of the abatement cost function with respect to
    mu """
    _theta1 = theta1(state, policy_state)
    _muy = PolicyState.muy(policy_state)
    _Theta_prime = _theta1 * theta2 * _muy**(theta2 - 1)
    return _Theta_prime

def Abatement(state, policy_state):
    """ Abatement cost value"""
    _Theta = Theta(state, policy_state)
    _ygross = ygross(state, policy_state)
    return _ygross * _Theta

def Omega(state, policy_state):
    """ Climate damage function """
    _TAT = State.TATx(state)
    _Omega = pi1 * _TAT**pow1 + pi2 * _TAT**pow2
    return _Omega

def Omega_prime(state, policy_state):
    """ The first derivative of the climate damage function """
    _TAT = State.TATx(state)
    _Omega_prime = pow1 * pi1 * _TAT**(pow1-1) + pow2 * pi2 * _TAT**(pow2-1)
    return _Omega_prime

def ygross(state, policy_state):
    """ Gross production in effective labor units """
    _kx = State.kx(state)  # Capital stock today
    _ygross = _kx**alpha
    return _ygross

def ynet(state, policy_state):
    """ Net production, where the climate damage is deducted, in effective
    labor units """
    _ynet = (1 - Omega(state, policy_state)) * ygross(state, policy_state)
    return _ynet

def Dam(state, policy_state):
    """ Damages """
    _dam =  Omega(state, policy_state) * ygross(state, policy_state)
    return _dam

def inv(state, policy_state):
    """ Investment """
    _con = con(state, policy_state)
    _ynet = ynet(state, policy_state)
    _Theta = Theta(state, policy_state)
    _inv = _ynet * (1 - _Theta) - _con
    return _inv


def Eind(state, policy_state):
    """ Industrial CO2 emission [1000 GtC] """
    _sigma = sigma(state, policy_state)
    _tfp = tfp(state, policy_state)
    _lab = lab(state, policy_state)
    _ygross = ygross(state, policy_state)
    _muy = PolicyState.muy(policy_state)
    _Eind = (1 - _muy) * _sigma * _ygross
    return _Eind

def Emissions(state, policy_state):
    """ Industrial CO2 emission [1000 GtC] """
    _tfp = tfp(state, policy_state)
    _lab = lab(state, policy_state)
    _Eind = Eind(state, policy_state) * _tfp * _lab
    _Eland = Eland(state, policy_state)
    return _Eind + _Eland

def carbontax(state, policy_state):
    _sigma = sigma(state, policy_state)
    _theta1 = theta1(state, policy_state)
    _muy = PolicyState.muy(policy_state)
    _carbontax = _theta1 * theta2 * _muy**(theta2 -1) /_sigma
    return _carbontax

def scc(state, policy_state):
    _lambd_haty = PolicyState.lambd_haty(policy_state)
    _Theta = Theta(state, policy_state)
    _Omega = Omega(state, policy_state)
    _nuAT_haty = PolicyState.nuAT_haty(policy_state)
    _muy = PolicyState.muy(policy_state)
    _kx = State.kx(state)
    _sigma = sigma(state, policy_state)
    _tfp = tfp(state, policy_state)
    _lab = lab(state, policy_state)
    _b12 = b12(state, policy_state)
    _nuUO_haty = PolicyState.nuUO_haty(policy_state)
    _etaAT_haty = PolicyState.etaAT_haty(policy_state)
    _c1 = c1(state, policy_state)
    _MATx = State.MATx(state)
    _dvdk =  (_lambd_haty * (Tstep * (1 - _Theta - _Omega) * alpha * _kx**(alpha-1)
                               + (1 - delta)**Tstep)
                    + (-_nuAT_haty) * (1 - _muy) * Tstep * _sigma * _tfp * _lab * alpha
                    * _kx**(alpha-1))

    _dvdMAT =  ((-_nuAT_haty) * (1 - _b12) + _nuUO_haty * _b12
                + _etaAT_haty * _c1 * f2xco2 / (tf.math.log(2.) * _MATx))
    _scc = - _dvdMAT / _dvdk * _tfp * _lab
    return _scc

# --------------------------------------------------------------------------- #
# State variables in period t+1
# --------------------------------------------------------------------------- #
def MATplus(state, policy_state):
    """ Carbon mass in the atmosphere """
    _tfp = tfp(state, policy_state)
    _lab = lab(state, policy_state)
    _sigma = sigma(state, policy_state)
    _Eland = Eland(state, policy_state)
    _muy = PolicyState.muy(policy_state)
    _kx = State.kx(state)
    _MATx = State.MATx(state)
    _MUOx = State.MUOx(state)
    _b21 = b21(state, policy_state)
    _b12 = b12(state, policy_state)
    _MATplus = (1-_b12) * _MATx + _b21 * _MUOx \
        + Tstep*(1 - _muy) * _sigma * _tfp * _lab * _kx**alpha + Tstep*_Eland
    return _MATplus


def MUOplus(state, policy_state):
    """ Carbon mass in the upper ocean """
    _MATx = State.MATx(state)
    _MUOx = State.MUOx(state)
    _MLOx = State.MLOx(state)
    _b21 = b21(state, policy_state)
    _b32 = b32(state, policy_state)
    _b12 = b12(state, policy_state)
    _b23 = b23(state, policy_state)
    _MUOplus = _b12 * _MATx + (1 - _b21 - _b23) * _MUOx + _b32 * _MLOx
    return _MUOplus


def MLOplus(state, policy_state):
    """ Carbon mass in the lower ocean """
    _MUOx = State.MUOx(state)
    _MLOx = State.MLOx(state)
    _b32 = b32(state, policy_state)
    _b23 = b23(state, policy_state)
    _MLOplus = _b23 * _MUOx + (1 - _b32) * _MLOx
    return _MLOplus


def TATplus(state, policy_state):
    """ Atmosphere temperature change relative to the preindustrial """
    _Fex = Fex(state, policy_state)
    _TATx = State.TATx(state)
    _TOCx = State.TOCx(state)
    _MATx = State.MATx(state)
    _c1c3 = c1c3(state, policy_state)
    _c1 = c1(state, policy_state)
    _c1f = c1f(state, policy_state)
    _TATplus = (1 - _c1c3 - _c1f) * _TATx + _c1c3 * _TOCx \
        + _c1 * (f2xco2 * (tf.math.log(_MATx / MATbase) / tf.math.log(2.))
                 + _Fex)
    return _TATplus


def TOCplus(state, policy_state):
    """ Ocean temperature change relative to the preindustrial """
    _TATx = State.TATx(state)
    _TOCx = State.TOCx(state)
    _c4 = c4(state, policy_state)
    _TOCplus = _c4 * _TATx + (1 - _c4) * _TOCx
    return _TOCplus
