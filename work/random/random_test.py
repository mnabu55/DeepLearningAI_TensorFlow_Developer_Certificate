import random
import numpy as np

seed = 42

rnd = np.random.RandomState(seed)
ar = rnd.randn(100 + 50)

print(ar)