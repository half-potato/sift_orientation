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
    octave = kp.octave & 255
    layer = (kp.octave >> 8) & 0xFF
    octave = octave if octave < 128 else (-128 | octave)
    kpts[i, 3] = layer
    kpts[i, 4] = octave
    kpts[i, 5] = kp.angle
    #  print(kp.octave, octave, layer)

_, inds = np.unique(kpts[:, 0]*1000 + kpts[:, 1], return_index=True)
#  print(inds)
kpts = kpts[inds, :]
print("Computing orientation")
maxOctave = int(np.max(kpts[:, 4]))
firstOctave = int(np.min(kpts[:, 4]))
actualNOctaves = maxOctave - firstOctave + 1
maxLayer = int(np.max(kpts[:, 3])) - 2
print(int(np.min(kpts[:, 4])))
print(f"Num Octaves: {actualNOctaves}, Num Layers: {maxLayer}, First Octave: {firstOctave}")
new_kpts = pbcvt.sift_desc(gray, kpts, firstOctave, actualNOctaves)

for i, kp in enumerate(kps[:20]):
    print(kpts[i], new_kpts[i])
