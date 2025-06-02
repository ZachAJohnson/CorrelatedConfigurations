# Zach Johnson May 2025
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from .utils import rs_from_n, n_from_rs

def Zbar_More(Z, num_density, T):
    """
    Finite Temperature Thomas Fermi Charge State using 
    R.M. More, "Pressure Ionization, Resonances, and the
    Continuity of Bound and Free States", Adv. in Atomic 
    Mol. Phys., Vol. 21, p. 332 (Table IV).
    
    Z = atomic number
    num_density = number density (1/cc)
    T = temperature (eV)
    """

    alpha = 14.3139
    beta = 0.6624
    a1 = 0.003323
    a2 = 0.9718
    a3 = 9.26148e-5
    a4 = 3.10165
    b0 = -1.7630
    b1 = 1.43175
    b2 = 0.31546
    c1 = -0.366667
    c2 = 0.983333
    
    convert = num_density*1.6726e-24
    R = convert/Z
    T0 = T/Z**(4./3.)
    Tf = T0/(1 + T0)
    A = a1*T0**a2 + a3*T0**a4
    B = -np.exp(b0 + b1*Tf + b2*Tf**7)
    C = c1*Tf + c2
    Q1 = A*R**B
    Q = (R**C + Q1**C)**(1/C)
    x = alpha*Q**beta

    return Z*x/(1 + x + np.sqrt(1 + 2.*x))

def Multispecies_Zbar_model(Z_array, species_densities_invcc, Te_eV):
	"""
	Multi-Scale Molecular Dynamics Model for Heterogeneous Charged Systems
		L. G. Stanton, J. N. Glosli and M. S. Murillo
		Physical Review X 8, 021044 (2018)
		https://github.com/MurilloGroupMSU/Thomas-Fermi-Multispecies-Ionization/tree/master
	"""  
	total_density_invcc = np.sum(species_densities_invcc)
	# Vector function to minimize to get effective species densities
	def f_to_make_zero(current_eff_densities_invcc):
	    current_Zbars = Zbar_More(Z_array, current_eff_densities_invcc, Te_eV)
	    ne = np.sum(species_densities_invcc * current_Zbars ) # single number
	    current_ne_array = current_eff_densities_invcc * current_Zbars # array that once solved should be all ne
	    vec_to_min = current_Zbars*current_eff_densities_invcc - ne
	    return vec_to_min
        
	initial_eff_densities_invcc = np.ones(len(species_densities_invcc))*total_density_invcc
	effective_densities_invcc = fsolve(f_to_make_zero, initial_eff_densities_invcc) # 1/(volume of single atom electron cloud)
	final_Zbars = Zbar_More(Z_array, effective_densities_invcc, Te_eV)
	return final_Zbars

def Zbar_func(Z, ni_invcc, Te_eV):
	if type(Z) in [int, float]:
		return Zbar_More(Z, ni_invcc, Te_eV)
	else:
		return Multispecies_Zbar_model(Z, ni_invcc, Te_eV)


if __name__ == '__main__':
	Te_eV = 1
	species_densities_invcc = np.array([2, 1, 0.5])*1e22 # in 1/cc
	species_Zs = np.array([1,6,8]) # nuclear charge not zbar

	zbars = Zbar_func(species_Zs, species_densities_invcc, Te_eV)

	print("Testing H, C, O")
	print(f"T_e = {Te_eV} [eV], densities {species_densities_invcc} [1/cc]")
	print(f"Resulting Zbar = {zbars}")