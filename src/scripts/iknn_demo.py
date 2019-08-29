import os
from pathlib import Path

import cv2
import nengo
import numpy as np

import itti_koch_niebur_nengo as iknn

img_path = str(Path(
    '../../data/visual_search_stimuli/'
    'ikkn_RVvGV/RVvGV/2/present/RVvGV_set_size_2_target_present_5.png')
)
img = cv2.imread(img_path)

S = iknn.visual_cortex.saliency.img_to_S(img)
S_flat = S.flatten()

model = nengo.Network()
with model:
    S_node = nengo.Node(output=S_flat)
    
    pulv_dims = S_node.size_out
    pulvinar = nengo.Ensemble(n_neurons=pulv_dims * 10, dimensions=pulv_dims)
    nengo.Connection(S_node, pulvinar)
