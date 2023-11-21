import numpy as np
from lib import Simulation as sim

p = np.array([
    [1, 0], 
    [0, 0]
    ])

q = np.array([
    [0, 0], 
    [0, 1]
    ])

sigma_x = np.array([
    [0, 1], 
    [1, 0]
    ])

sigma_y = np.array([
    [0, -1j], 
    [1j, 0]
    ])

sigma_z = np.array([
    [1, 0], 
    [0, -1]
    ])

n_sites = 2
theta = np.pi / 6 
phi = 2 * np.pi / 3 
beta = 0

fsim = np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta), 1j * np.exp(1j * beta) * np.sin(theta), 0],
    [0, 1j * np.exp(-1j * beta) * np.sin(theta), np.cos(theta), 0],
    [0, 0, 0, np.exp(1j * phi)]
], dtype=complex)


middle_matrix = np.identity(2 ** (n_sites - 2))

# Assuming beta is 0
m_1 = np.kron(np.kron(p, middle_matrix), p)
m_2 = np.exp(1j * phi) * np.kron(np.kron(q, middle_matrix), q)
m_3 = 0.5 * np.cos(theta) * (np.kron(np.kron(np.identity(2), middle_matrix), np.identity(2)) - np.kron(np.kron(sigma_z, middle_matrix), sigma_z))
m_4 = 0.5 * 1j * np.sin(theta) * (np.kron(np.kron(sigma_x, middle_matrix), sigma_x) + np.kron(np.kron(sigma_y, middle_matrix), sigma_y))

total_matrix = m_1 + m_2 + m_3 + m_4

print(fsim)
print("\n \n \n \n")
print(total_matrix)

print("\n \n \n \n")
print(sim.check_unitary(fsim))
print(sim.check_unitary(total_matrix))