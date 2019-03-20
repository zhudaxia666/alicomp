import numpy as np
from keras.utils import to_categorical

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11,16,0]
data = np.array(data)
print(data)

one_hots = to_categorical(data)
print(one_hots)

