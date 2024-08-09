import numpy as np
from cleanNewProject.config import Modelvariables
from cleanNewProject.Kernels.k import k
from cleanNewProject.Kernels.DkDy import DkDy
from cleanNewProject.Kernels.DkDx import DkDx
from cleanNewProject.Kernels.DkDxDy import DkDxDy
from cleanNewProject.Kernels.D2kDy2 import D2kDy2
from cleanNewProject.Kernels.D2kDx2 import D2kDx2
from cleanNewProject.Kernels.D3kDx2Dy import D3kDx2Dy
from cleanNewProject.Kernels.D3kDxDy2 import D3kDxDy2
from cleanNewProject.Kernels.D4kDx2Dy2 import D4kDx2Dy2


# Define the parameters
param1 = np.array([-0.3])
param2 = np.array([0.1])
param3 = np.array([0, 0])
param4 = 1

# Call the functions
a = DkDy(param1, param2, param3, param4)
b = DkDx(param1, param2, param3, param4)
c = DkDxDy(param1, param2, param3, param4)
d = D2kDx2(param1, param2, param3, param4)
e = D2kDy2(param1, param2, param3, param4)
f = D3kDx2Dy(param1, param2, param3, param4)
g = D3kDxDy2(param1, param2, param3, param4)
h = D4kDx2Dy2(param1, param2, param3, param4)
j = k(param1, param2, param3, param4)

# Display the results
print(f'DkDy: {a}')
print(f'DkDx: {b}')
print(f'DkDxDy: {c}')
print(f'D2kDx2: {d}')
print(f'D2kDy2: {e}')
print(f'D3kDx2Dy: {f}')
print(f'D3kDxDy2: {g}')
print(f'D4kDx2Dy2: {h}')
print(f'k: {j}')