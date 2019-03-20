import h5py
import numpy as np
import os
# base_dir = "data"
# path_training = os.path.join(base_dir, "training.h5")
f=h5py.File("new_train1.h5",'r')
label=f["label"]
print(np.sum(label,axis=0))