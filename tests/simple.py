import cv2
import numpy as np
import sift_ori
from pathlib import Path
import torch.nn.functional as F
import torch

path = str(Path(__file__).parent / "../assets/test.png")
print(path)
image = cv2.imread(path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Detecting")
sift = cv2.SIFT_create()
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
print(f"Num Octaves: {actualNOctaves}, Num Layers: {maxLayer}, First Octave: {firstOctave}")

# Try to compute scale
# First, construct the image pyramid
# Write function to do this
# Then, perform a search over the image pyramid so we can find the right size
# First, print the values across the image pyramid at the location
# I want to know if there is some obvious pattern
# Then I can try to compute some kind of neighborliness using a center surround kernel
ret = sift_ori.dog_pyramid(gray, firstOctave, actualNOctaves)
values = []
H, W = gray.shape
N = kpts.shape[0]
for layer in ret:
    hr, wr = layer.shape[-2:]
    grid = np.zeros((N, 2))
    grid[:, 0] = kpts[:, 0]/W*2 - 1
    grid[:, 1] = kpts[:, 1]/H*2 - 1
    grid = torch.tensor(grid).view(1, 1, N, 2).float()
    layer_th = torch.tensor(layer).view(1, -1, hr, wr).float()
    vals = F.grid_sample(layer_th, grid, mode="bilinear", align_corners=False).view(-1)
    values.append(vals)
    # scipy grid_sample per a layer to retrieve a list of values for each keypoints
values = torch.stack(values, dim=0)
print(values)
#  print(ret[0])
#  print(ret[1])

new_kpts = sift_ori.sift_desc(gray, kpts, firstOctave, actualNOctaves)

#  for i, kp in enumerate(kps[:20]):
#      print(kpts[i], new_kpts[i])
