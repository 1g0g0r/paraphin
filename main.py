import numpy as np

from my_consatnts import *
from my_mesh import create_mesh
from calc_p import calc_p


points, cells = create_mesh()

calc_p(cells)
