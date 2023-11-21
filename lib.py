import numpy as np
import math

class Simulation:

    def __init__(self, theta = np.pi / 6, phi = 2 * np.pi / 3, beta = 0, n_excited = 1, n_sites = 24):
        self._theta = theta # The amplitude representing kinetic / hopping energy in radians
        self._phi = phi # The phase angle representing interaction strength in radians 
        self._beta = beta # The phase factor representing magnetic flux in radians
        self.__n_excited = n_excited # The number of excited qbits
        self.__n_sites = n_sites # Has to be larger than 2
        
        if self.__n_excited > self.__n_sites:
            raise Exception("You cannot have more excited states then there are sites")

        if self.__n_sites < 2:
            raise Exception("You cannot have less than three sites")

        if self.__n_excited < 0:
            raise Exception("You cannot have a negative number of excited states")
        
        self.__initial_state = self.__create_initial_state()
        self.__current_state = self.__initial_state

        self.__unitary_matrix = self.__create_unitary_matrix()


    def get_initial_state(self): 
        return self.__initial_state

    def get_current_state(self): 
        return self.__current_state

    def get_unitary_matrix(self):
        return self.__unitary_matrix

    def apply_cycle(self):
        self.__current_state = np.matmul(self.__unitary_matrix, self.__current_state)

    def __create_initial_state(self):
        state_zero = np.array([1, 0])
        state_one = np.array([0, 1])

        # Creates a circuit of n_sites qbit sites with a ground state in each, remember for example [14] is actually site 15
        self._circuit = np.full((self.__n_sites, *state_zero.shape), state_zero)

        array_mid_point = math.ceil(self.__n_sites / 2) - 1
        odd = (self.__n_sites % 2)
        # Puts n qbit excited states in the centre
        start = (array_mid_point - math.floor((self.__n_excited - 1 + odd) / 2))
        end = (array_mid_point + 1 + math.ceil((self.__n_excited - 1 - odd) / 2))
        for i in range(start, end):
            self._circuit[i] = state_one

        # Does the tensor product of the whole space
        state = self._circuit[0]
        for i in range(1, self.__n_sites):
            state = np.kron(state, self._circuit[i])

        return state 

    def __create_unitary_matrix(self):

        # Sets up matrices we need to use
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

        # Creates an sim gate that acts on two qbits using the inputted parameters 
        fsim = np.array([
            [1, 0, 0, 0],
            [0, np.cos(self._theta), 1j * np.exp(1j * self._beta) * np.sin(self._theta), 0],
            [0, 1j * np.exp(-1j * self._beta) * np.sin(self._theta), np.cos(self._theta), 0],
            [0, 0, 0, np.exp(1j * self._phi)]
        ], dtype=complex)
        
        # if self.check_unitary(fsim) == False:
        #     raise Exception("The fsim is not unitary")
        # IS GOOD

        odd_matrix = np.identity(2 ** (self.__n_sites))
        even_matrix = np.identity(2 ** (self.__n_sites))

        # i is the site (starting at zero) where the fsim gate acts on
        for i in range(0, self.__n_sites):
            # For the case when the fsim is acting on the edges 
            if i == self.__n_sites - 1:
                # Get the matrix in the middle as such
                middle_matrix = np.identity(2 ** (self.__n_sites - 2))

                # Assuming beta is 0
                m_1 = np.kron(np.kron(p, middle_matrix), p)
                m_2 = np.exp(1j * self._phi) * np.kron(np.kron(q, middle_matrix), q)
                m_3 = 0.5 * np.cos(self._theta) * (np.kron(np.kron(np.identity(2), middle_matrix), np.identity(2)) - np.kron(np.kron(sigma_z, middle_matrix), sigma_z))
                m_4 = 0.5 * 1j * np.sin(self._theta) * (np.kron(np.kron(sigma_x, middle_matrix), sigma_x) + np.kron(np.kron(sigma_y, middle_matrix), sigma_y))

                total_matrix = m_1 + m_2 + m_3 + m_4

                # if self.check_unitary(total_matrix) == False:
                #     raise Exception("The total_matrix is not unitary") 
                # IS GOOD

            else: # For every other case 
                # Get the matrix before where the fsim gate is applied
                before_matrix = np.identity(2 ** i)

                # Get the matrix after where the fsim gate is applied
                after_matrix = np.identity(2 ** (self.__n_sites - i -2))

                # Tensors the before, fsim and after matrices together in that order
                total_matrix = np.kron(np.kron(before_matrix, fsim), after_matrix)

                # if self.check_unitary(total_matrix) == False:
                #   raise Exception("The total_matrix is not unitary")
                # IS GOOD

            if i % 2 == 0: # If even
                even_matrix = np.matmul(even_matrix, total_matrix)
            else: # If odd
                odd_matrix = np.matmul(odd_matrix, total_matrix)

        unitary_matrix = np.matmul(odd_matrix, even_matrix)
        # if self.check_unitary(total_matrix) == False:
        #   raise Exception("The total_matrix is not unitary")
        # IS GOOD

        return unitary_matrix

    @staticmethod
    def check_unitary(matrix): # Works but for medium n takes ages 
        if matrix.shape[0] != matrix.shape[1]: # Check if square matrix
            return False
        
        # Find hermitian conjugate 
        matrix_t = np.transpose(matrix)
        matrix_h = np.conjugate(matrix_t) 

        result = np.matmul(matrix_h, matrix) # Multiply hermitian by original

        result = result.round(10) # Round each element to 10 decimal places to account for floating point errors
        # Check if real and 
        size = result.shape[0] 
        if np.all(np.imag(result) == 0) and np.allclose(result, np.identity(size)):
            return True
        return False 
            


