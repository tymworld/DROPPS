from math import sqrt

# Parameters for default bond
bond_length = 0.38

# Parameters for LJ interaction
epsilon = 0.2 * 4.1868

# Parameters for Coulomb interaction
relative_permittivity = 80
k0 = 138.50895744
kappa_epsilon_0 = 8.8541878128 * pow(10, -12)
kappa_R = 8.31446261815324
kappa_F = 96485.3321233100184
kappa_coefficient = sqrt(kappa_epsilon_0 * kappa_R / (2000 * pow(kappa_F, 2))) * pow(10, 9)