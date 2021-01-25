import cv2
import numpy as np
import pbcvt # your module, also the name of your compiled dynamic library file w/o the extension
from pathlib import Path

path = str(Path(__file__).parent / "../assets/test.png")
print(path)
image = cv2.imread(path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Detecting")
sift = cv2.xfeatures2d.SIFT_create()
kps = sift.detect(gray, None)
kpts = np.zeros((len(kps), 6), dtype=np.float32)
for i, kp in enumerate(kps):
    kpts[i, 0] = kp.pt[0]
    kpts[i, 1] = kp.pt[1]
    kpts[i, 2] = kp.size
    print(f"{kp.octave:b}")
    octave = kp.octave & 255
    layer = (kp.octave >> 8) & 0xFF
    octave = octave if octave < 128 else (-128 | octave)
    print(octave, layer)
    kpts[i, 3] = layer # welp
    kpts[i, 4] = octave + 1

print(kpts)
print("Computing orientation")
pbcvt.sift_desc(gray, kpts, 1+int(np.max(kpts[:, 4])))

print(kpts)
for i, kp in enumerate(kps):
    print(kpts[i, 5], kp.angle)
