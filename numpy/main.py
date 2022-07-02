
import numpy as np


a = np.arange(6).reshape(2,3) + 10
# a
# array([[10, 11, 12],
#        [13, 14, 15]])
np.argmax(a)
# 5
np.argmax(a, axis=0)
# array([1, 1, 1])
np.argmax(a, axis=1)
# array([2, 2])
