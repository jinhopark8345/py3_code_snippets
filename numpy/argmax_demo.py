from PIL import Image
import numpy as np


# a = np.arange(6).reshape(2, 3) + 10

a = np.array([
    [6, 4, 1],
    [2, 3, 5]
])


print(np.argmax(a, axis=0)) # ([0, 0, 1]), # in each column, biggest number's index
print(np.argmax(a, axis=1)) # ([0, 2]), # in each row, biggest number's index

tmp = np.array([
    [1, 1, 0],
    [0, 0, 1]
])

# print(tmp)

# return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
print(f"{ tmp = }")
print(f"{ np.argmax(tmp, axis=0) = }")
print(f"{ np.argmax(tmp, axis=0) * 255 = }")
print(f"{ np.argmax(tmp, axis=0) * 255 / tmp.shape[0] = }")
print(f"{ (np.argmax(tmp, axis=0) * 255 / tmp.shape[0]).astype(np.uint8) = }")
print(
    f"{ Image.fromarray((np.argmax(tmp, axis=0) * 255 / tmp.shape[0]).astype(np.uint8)) = }"
)
