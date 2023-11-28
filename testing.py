import numpy as np

p0 = np.array([
    [1, 0], 
    [0, 0]
    ])

p1 = np.array([
    [0, 0], 
    [0, 1]
    ])

result = np.kron(np.kron(p1, p1), p0)
print(result)