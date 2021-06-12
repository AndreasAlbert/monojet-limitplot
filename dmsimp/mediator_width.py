"""
Collection of functions for the calculation of DMsimp mediator width.

All functions are obtained from the LHC DMWG report:

https://arxiv.org/abs/1703.05703

Please cite accordingly.
"""


from __future__ import division
import math
import numpy as np

# Quark and lepton masses are constant inputs
m_quarks = [0.0024, 0.0048, 0.104, 1.27, 4.2, 171.2]
m_leptons = [0.000511, 0.105658, 1.77682]


def beta(m_med, m_f):
    """Convenience function that implements part of the spin-1 width formula.

    :param m_med: mediator mass
    :type m_med: float
    :param m_f: fermion mass
    :type m_f: float
    """
    return np.sqrt(1 - 4 * m_f**2 / m_med**2)


def alpha(m_med, m_f):
    """Convenience function that implements part of the spin-1 width formula.

    :param m_med: mediator mass
    :type m_med: float
    :param m_f: fermion mass
    :type m_f: float
    """
    return 1 + 2 * m_f**2 / m_med**2


# AXIAL
def gamma_axial_lepton(m_med, g_l=0.0):
    """Function to calculate charged lepton width of an axial vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_l: lepton coupling, defaults to 0.0
    :type g_l: float, optional
    """
    gamma = 0
    for m_l in m_leptons:
        if(m_med >= 2 * m_l):
            gamma = gamma + g_l**2 * m_med / \
                (12 * math.pi) * beta(m_med, m_l)**3
    return gamma


def gamma_axial_neutrino(m_med, g_l=0.0):
    """Function to calculate neutrino width of an axial vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_l: lepton coupling, defaults to 0.0
    :type g_l: float, optional
    """
    return 3 * g_l**2 / (24 * math.pi) * m_med


def gamma_axial_quark(m_med, g_q=0.25):
    """Function to calculate quark width of an axial vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_q: quark coupling, defaults to 0.25
    :type g_q: float, optional
    """
    gamma = 0
    for m_q in m_quarks:
        if(m_med >= 2 * m_q):
            gamma = gamma + 3 * g_q**2 * m_med / \
                (12 * math.pi) * beta(m_med, m_q)**3
    return gamma


def gamma_axial_chi(m_med, m_x, g_chi=1.0):
    """Function to calculate DM width of an axial vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param m_x: DM mass
    :type m_x: float
    :param g_chi: quark coupling, defaults to 0.25
    :type g_chi: float, optional
    """
    gamma = 0
    if(m_med >= 2 * m_x):
        gamma = gamma + g_chi**2 * m_med / \
            (12 * math.pi) * beta(m_med, m_x) ** 3
    return gamma


def gamma_axial_total(m_med, m_x, g_q=0.25, g_chi=1.0, g_l=0.0):
    """Function to calculate total width of an axial vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param m_x: DM mass
    :type m_x: float
    :param g_q: quark coupling, defaults to 0.25
    :type g_q: float, optional
    :param g_chi: DM coupling, defaults to 0.25
    :type g_chi: float, optional
    :param g_l: lepton coupling, defaults to 0.0
    :type g_l: float, optional
    """
    gamma = 0
    gamma = gamma + gamma_axial_lepton(m_med, g_l)
    gamma = gamma + gamma_axial_neutrino(m_med, g_l)
    gamma = gamma + gamma_axial_quark(m_med, g_q)
    gamma = gamma + gamma_axial_chi(m_med, m_x, g_chi)
    return gamma


# VECTOR
def gamma_vector_lepton(m_med, g_l=0.0):
    """Function to calculate lepton width of a vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_l: lepton coupling, defaults to 0.0
    :type g_l: float, optional
    """
    gamma = 0
    for m_l in m_leptons:
        if(m_med >= 2 * m_l):
            gamma = gamma + g_l**2 * m_med / \
                (12 * math.pi) * alpha(m_med, m_l) * beta(m_med, m_l)
    return gamma


def gamma_vector_neutrino(m_med, g_l=0.0):
    """Function to calculate the neutrino width of a vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_l: lepton coupling, defaults to 0.0
    :type g_l: float, optional
    """
    return 3 * g_l**2 / (24 * math.pi) * m_med


def gamma_vector_quark(m_med, g_q=0.25):
    """Function to calculate quark width of an axial vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_q: quark coupling, defaults to 0.25
    :type g_q: float, optional
    """
    gamma = 0
    for m_q in m_quarks:
        if(m_med >= 2 * m_q):
            gamma = gamma + 3 * g_q**2 * m_med / \
                (12 * math.pi) * alpha(m_med, m_q) * beta(m_med, m_q)
    return gamma


def gamma_vector_chi(m_med, m_x, g_chi=1.0):
    """Function to calculate DM width of a vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param m_x: DM mass
    :type m_x: float
    :param g_q: quark coupling, defaults to 0.25
    :type g_q: float, optional
    """
    gamma = 0
    if(m_med >= 2 * m_x):
        gamma = gamma + g_chi**2 * m_med / \
            (12 * math.pi) * alpha(m_med, m_x) * beta(m_med, m_x)
    return gamma


def gamma_vector_total(m_med, m_x, g_q=0.25, g_chi=1.0, g_l=0.0):
    """Function to calculate total width of a  vector mediator

    :param m_med: mediator mass
    :type m_med: float
    :param m_x: DM mass
    :type m_x: float
    :param g_q: quark coupling, defaults to 0.25
    :type g_q: float, optional
    :param g_chi: DM coupling, defaults to 0.25
    :type g_chi: float, optional
    :param g_l: lepton coupling, defaults to 0.0
    :type g_l: float, optional
    """
    gamma = 0
    gamma = gamma + gamma_vector_lepton(m_med, g_l)
    gamma = gamma + gamma_vector_neutrino(m_med, g_l)
    gamma = gamma + gamma_vector_quark(m_med, g_q)
    gamma = gamma + gamma_vector_chi(m_med, m_x, g_chi)
    return gamma


def gamma_pseudo_chi(m_med, m_x, g_chi):
    """Function to calculate DM width of a pseudoscalar mediator

    :param m_med: mediator mass
    :type m_med: float
    :param m_x: DM mass
    :type m_x: float
    :param g_q: quark coupling, defaults to 0.25
    :type g_q: float, optional
    """
    if(m_x >= m_med * 0.5):
        return 0
    gamma = g_chi**2 * m_med / (8 * math.pi) * beta(m_med, m_x)
    return gamma


def gamma_pseudo_quark(m_med, g_q):
    """Function to calculate quark width of a pseudoscalar mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_q: quark coupling, defaults to 0.25
    :type g_q: float, optional
    """
    gamma = 0
    for m_q in m_quarks:
        if(m_med >= 2 * m_q):
            y_q = np.sqrt(2) * m_q / 246
            gamma += 3 * g_q**2 * y_q**2 * m_med / \
                (16 * math.pi) * beta(m_med, m_q)
    return gamma


def gamma_pseudo_gluon(m_med, g_q):
    """Function to calculate gluon width of a pseudoscalar mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_q: quark coupling
    :type g_q: float
    """
    alpha_s = 0.130
    v = 246
    gamma = alpha_s ** 2 * g_q**2 * m_med**3 / (32 * math.pi**3 * v**2)
    gamma = gamma * np.abs(f_ps(4 * (m_quarks[5] / m_med)**2))**2
    return gamma


def gamma_pseudo_total(m_med, m_x, g_q, g_chi):
    """Function to calculate total width of a pseudoscalar mediator

    :param m_med: mediator mass
    :type m_med: float
    :param m_x: DM mass
    :type m_x: float
    :param g_q: quark coupling, defaults to 0.25
    :type g_q: float, optional
    :param g_chi: DM coupling, defaults to 0.25
    :type g_chi: float, optional
    """
    gamma = 0
    gamma += gamma_pseudo_chi(m_med=m_med, m_x=m_x, g_chi=g_chi)
    gamma += gamma_pseudo_quark(m_med=m_med, g_q=g_q)
    gamma += gamma_pseudo_gluon(m_med=m_med, g_q=g_q)
    return gamma


def gamma_scalar_chi(m_med, m_x, g_chi):
    """Function to calculate DM width of a scalar mediator

    :param m_med: mediator mass
    :type m_med: float
    :param m_x: DM mass
    :type m_x: float
    :param g_q: quark coupling
    :type g_q: float
    """
    if(m_x >= m_med * 0.5):
        return 0
    gamma = g_chi**2 * m_med / (8 * math.pi) * beta(m_med, m_x) ** 3
    return gamma


def gamma_scalar_quark(m_med, g_q):
    """Function to calculate quark width of a scalar mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_q: quark coupling
    :type g_q: float
    """
    gamma = 0
    for m_q in m_quarks:
        if(m_med >= 2 * m_q):
            y_q = np.sqrt(2) * m_q / 246
            gamma += 3 * g_q**2 * y_q**2 * m_med / \
                (16 * math.pi) * beta(m_med, m_q) ** 3
    return gamma


def gamma_scalar_gluon(m_med, g_q):
    """Function to calculate gluon width of a scalar mediator

    :param m_med: mediator mass
    :type m_med: float
    :param g_q: quark coupling
    :type g_q: float
    """
    alpha_s = 0.130
    v = 246
    gamma = alpha_s ** 2 * g_q**2 * m_med**3 / (32 * math.pi**3 * v**2)
    gamma = gamma * np.abs(f_s(4 * (m_quarks[5] / m_med)**2))**2
    return gamma


def gamma_scalar_total(m_med, m_x, g_q, g_chi):
    """Function to calculate total width of a scalar mediator

    :param m_med: mediator mass
    :type m_med: float
    :param m_x: DM mass
    :type m_x: float
    :param g_q: quark coupling, defaults to 0.25
    :type g_q: float, optional
    :param g_chi: DM coupling, defaults to 0.25
    :type g_chi: float, optional
    """
    gamma = 0
    gamma += gamma_scalar_chi(m_med=m_med, m_x=m_x, g_chi=g_chi)
    gamma += gamma_scalar_quark(m_med=m_med, g_q=g_q)
    gamma += gamma_scalar_gluon(m_med=m_med, g_q=g_q)
    return gamma


def f_ps(tau):
    """Convenience function that implements part of the spin-0 width formula.

    :param m_med: mediator mass
    :type m_med: float
    :param m_f: dimensionless parameter
    :type m_f: float
    """
    tau = tau.astype(np.complex) #np.complex(tau, 0)
    return tau * (np.arctan(1. / np.sqrt(tau - 1)))**2


def f_s(tau):
    """Convenience function that implements part of the spin-0 width formula.

    :param m_med: mediator mass
    :type m_med: float
    :param m_f: dimensionless parameter
    :type m_f: float
    """
    tau = np.complex(tau, 0)
    return tau * (1 + (1 - tau) * (np.arctan(1. / np.sqrt(tau - 1)))**2)
