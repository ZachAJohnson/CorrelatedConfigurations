import numpy as np
from .constants import *


def n_from_rs( rs):
    """
    Sphere radius to density, in any units
    """
    return 1/(4/3*π*rs**3)

def rs_from_n(n):
    """
    Density to sphere radius, in any units.
    """
    return (4/3*π*n)**(-1/3)

def Debye_length(T, ni, Zbar):
    """
    Inputs in a.u.
    """
    ne = Zbar*ni
    EF = Fermi_Energy(ne)
    T_effective = (T**1.8 + (2/3*EF)**1.8)**(1/1.8)
    λD = 1/np.sqrt(  4*π*ne/T_effective ) # Including degeneracy
    # λD = 1/np.sqrt(  4*π*ne/T_effective + 4*π*Zbar**2*ni/T  )  # Including ions
    return λD

def Kappa(T, ni, Zbar):
    """
    Inputs in a.u.
    """
    rs = rs_from_n(ni)
    λD = Debye_length(T, ni, Zbar) # ne = ni Zbar = sum_j n_j Zbar_j => Zbar = sum_j n_j Zbar_j/sum_j n_j for multiple species
    return rs/λD

def Fermi_Energy(ne):
    E_F = 1/(2*m_e) * (3*π**2 * ne)**(2/3)
    return E_F