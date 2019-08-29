import nengo
import numpy as np


def pulvinar(size_in, net=None):
    """"""
    if net is None:
        net = nengo.Network()

    with net:
        S_input = nengo.Node(size_in=size_in, label='S_input')
        # Ensemble with 100 LIF neurons which represents a 2-dimensional signal
        pulv = nengo.Ensemble(100, dimensions=size_in, max_rates=Uniform(100, 200))

        pulv = nengo.Ensemble(n_neurons=size_in, label='pulvinar')
        nengo.Config(S_input, pulv.neurons, transform=np.ones())