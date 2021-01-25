import cv2
import numpy as np
import pbcvt # your module, also the name of your compiled dynamic library file w/o the extension

a = np.array([[1., 2., 3.]])
kpts = np.random.rand(8, 3)
print(pbcvt.sift_desc(a, kpts, 3))
