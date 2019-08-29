import cv2
import nengo
import numpy as np

import itti_koch_niebur_nengo as iknn

toy_S = np.array([1,2,3,4]).reshape([2,2])
toy_S_flat = toy_S.flatten()

model = nengo.Network()
with model:
    S_node = nengo.Node(output=toy_S_flat)
    
    pulv_dims = S_node.size_out
    pulvinar = nengo.Ensemble(n_neurons=100, dimensions=pulv_dims)
    nengo.Connection(S_node, pulvinar)
    
    wta = iknn.fef.LCA(dimensions=pulv_dims, n_neurons=100)
    nengo.Connection(pulvinar, wta.input)

    wta_max = nengo.Ensemble(n_neurons=100, dimensions=1, radius=10)
    wta_max_c = nengo.Connection(wta.output, wta_max, function=np.argmax)

    wta_round = nengo.Ensemble(n_neurons=100, dimensions=1, radius=10)
    wta_round_c = nengo.Connection(wta_max, wta_round, 
                                   function= lambda x: int(np.round(x))
                                  )