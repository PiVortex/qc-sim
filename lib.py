import numpy as np
import math

class Simulation:

    def __init__(self, theta = np.pi / 6, phi = 2 * np.pi / 3, beta = 0, n_excited = 1, n_sites = 24, debug = False):
        self._theta = theta # The amplitude representing kinetic / hopping energy in radians
        self._phi = phi # The phase angle representing interaction strength in radians 
        self._beta = beta # The phase factor representing magnetic flux in radians
        self._n_excited = n_excited # The number of excited qbits
        self._n_sites = n_sites # Has to be larger than 2, should be even
        self.debug = debug
        
        if self._n_excited > self._n_sites:
            raise Exception("You cannot have more excited states then there are sites")

        if self._n_sites < 2:
            raise Exception("You cannot have less than three sites")

        if self._n_excited < 0:
            raise Exception("You cannot have a negative number of excited states")
        
        self.__initial_state = self.__create_initial_state()
        self.__current_state = self.__initial_state

        self.__unitary_matrix = self.__create_unitary_matrix()

        self.__obs_matrix = self.__create_obs_matrix()

    def get_initial_state(self): 
        return self.__initial_state

    def get_current_state(self): 
        return self.__current_state

    def get_unitary_matrix(self):
        return self.__unitary_matrix

    def get_obs_matrix(self):
        return self.__obs_matrix

    def apply_cycle(self):
        self.__current_state = np.matmul(self.__unitary_matrix, self.__current_state)
        if self.debug == True:
            if self.check_norm(self.__current_state) == False:
                raise Exception("The vector is not normalised")

    def apply_obs(self):
        state_t = np.transpose(self.__current_state)
        state_h = np.conjugate(state_t) 
        
        probability = np.dot(state_h, np.dot(self.__obs_matrix, self.__current_state))
        
        return np.real(probability)

    def __create_initial_state(self):
        state_zero = np.array([1, 0])
        state_one = np.array([0, 1])

        # Creates a circuit of n_sites qbit sites with a ground state in each, remember for example [4] is actually site 5
        self._circuit = np.zeros(self._n_sites)
        # self._circuit = np.full((self._n_sites, *state_zero.shape), state_zero)

        array_mid_point = math.ceil(self._n_sites / 2) - 1
        odd = (self._n_sites % 2)
        # Puts n qbit excited states in the centre
        start = (array_mid_point - math.floor((self._n_excited - 1 + odd) / 2))
        end = (array_mid_point + 1 + math.ceil((self._n_excited - 1 - odd) / 2))
        for i in range(start, end):
            self._circuit[i] = 1

        # Does the tensor product of the whole space
        state = 1
        for i in range(0, self._n_sites):
            if self._circuit[i] == 0:
                state = np.kron(state, state_zero)
            elif self._circuit[i] == 1:
                state = np.kron(state, state_one)
            else:
                raise Exception("Circuit error")

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
        
        if self.debug == True:
            if self.check_unitary(fsim) == False:
                raise Exception("The fsim is not unitary")

        dim = 2 ** self._n_sites
        odd_matrix = np.identity(dim)
        even_matrix = np.identity(dim)

        # i is the site (starting at zero) where the fsim gate acts on
        for i in range(0, self._n_sites):
            # For the case when the fsim is acting on the edges 
            if i == self._n_sites - 1:
                # Get the matrix in the middle as such
                middle_matrix = np.identity(2 ** (self._n_sites - 2))

                # Assuming beta is 0
                m_1 = np.kron(np.kron(p, middle_matrix), p)
                m_2 = np.exp(1j * self._phi) * np.kron(np.kron(q, middle_matrix), q)
                m_3 = 0.5 * np.cos(self._theta) * (np.kron(np.kron(np.identity(2), middle_matrix), np.identity(2)) - np.kron(np.kron(sigma_z, middle_matrix), sigma_z))
                m_4 = 0.5 * 1j * np.sin(self._theta) * (np.kron(np.kron(sigma_x, middle_matrix), sigma_x) + np.kron(np.kron(sigma_y, middle_matrix), sigma_y))

                sub_matrix = m_1 + m_2 + m_3 + m_4

                if self.debug == True: 
                    if self.check_unitary(sub_matrix) == False:
                        raise Exception("The edge sub_matrix is not unitary") 

            else: # For every other case 
                # Get the matrix before where the fsim gate is applied
                before_matrix = np.identity(2 ** i)

                # Get the matrix after where the fsim gate is applied
                after_matrix = np.identity(2 ** (self._n_sites - i -2))

                # Tensors the before, fsim and after matrices together in that order
                sub_matrix = np.kron(np.kron(before_matrix, fsim), after_matrix)

                if self.debug == True:
                    if self.check_unitary(sub_matrix) == False:
                        raise Exception("The normal sub_matrix is not unitary")

            if i % 2 == 0: # If even
                even_matrix = np.matmul(even_matrix, sub_matrix)
            else: # If odd
                odd_matrix = np.matmul(odd_matrix, sub_matrix)

        unitary_matrix = np.matmul(odd_matrix, even_matrix)

        if self.debug == True: 
            if self.check_unitary(unitary_matrix) == False:
                raise Exception("The unitary_matrix is not unitary")

        return unitary_matrix

    def __create_obs_matrix(self):

        dim = 2 ** self._n_sites
        obs_matrix = np.zeros((dim, dim))
        # Consider all excited, n_sites = n_excited? Might already be handled
        for i in range(0, self._n_sites):
            counter = self._n_sites - self._n_excited - i
            if counter >= 0:
                bit_string = i * "0" + self._n_excited * "1" + counter * "0"
            else: 
                bit_string = (0 - counter) * "1" + (i + counter) * "0" + (self._n_excited + counter) * "1"

            index = int(bit_string, 2)
            obs_matrix[index, index] = 1

        return obs_matrix

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

    @staticmethod
    def check_norm(vector):
        vector_t = np.transpose(vector)
        vector_h = np.conjugate(vector_t) 
        
        dot = np.vdot(vector_t, vector)
        dot = dot.round(10)
        if np.imag(dot) == 0 and np.real(dot) == 1:
            return True
        return False 
