"""winner-take-all networks
from <https://github.com/ctn-waterloo/cogsci17-decide>
"""
import nengo
import numpy as np


def LCA(dimensions, n_neurons, k=1., beta=1., tau_model=0.1,
        tau_actual=0.1, add_lambda_out=True):
    """

    Parameters
    ----------
    dimensions
    n_neurons
    k
    beta
    tau_model
    tau_actual
    add_lambda_out : bool
        if True, add following to output node:
            output=lambda t, x: x
        (hack to make not a passthrough node)

    Returns
    -------
    LCA : nengo.Network
    """
    # eqn (4) ignoring truncation, put into continuous LTI form:
    #   dot{x} = Ax + Bu
    I = np.eye(dimensions)
    inhibit = 1 - I
    B = 1. / tau_model
    A = (-k * I - beta * inhibit) / tau_model

    with nengo.Network(label="LCA") as net:
        net.input = nengo.Node(size_in=dimensions)
        x = nengo.networks.EnsembleArray(
            n_neurons, dimensions,
            eval_points=nengo.dists.Uniform(0., 1.),
            intercepts=nengo.dists.Uniform(0., 1.),
            encoders=nengo.dists.Choice([[1.]]), label="state")
        nengo.Connection(x.output, x.input, transform=tau_actual * A + I,
                         synapse=tau_actual)

        nengo.Connection(
            net.input, x.input,
            transform=tau_actual * B,
            synapse=tau_actual)

        if add_lambda_out:
            lambda_out = nengo.Node(size_in=x.output.size_out,
                                    output=lambda t, x: x)
            nengo.Connection(x.output, lambda_out)
            net.output = lambda_out
        else:
            net.output = x.output

    return net


def IA(d, n_neurons, dt, share_thresholding_intercepts=False):
    bar_beta = 2.  # should be >= 1 + max_input * tau2 / tau1
    tau_model1 = 0.1
    tau_model2 = 0.1
    tau_actual = 0.1

    # dynamics put into continuous LTI form:
    #   dot{x1} = A1x1 + A2x2 + Bu
    # where x1 is the state variable for layer 1 and
    #       x2 is the state variable for layer 2
    # note that from the perspective of Principle 3, A2x2 is treated
    # as an "input" similar to u
    I = np.eye(d)
    inhibit = 1 - I
    B = 1. / tau_model1  # input -> layer 1
    A1 = 0  # (integrator) layer1 -> layer1
    A2 = (I - bar_beta * inhibit) / tau_model2  # layer 2 -> layer 1

    n_neurons_threshold = 50
    n_neurons_x = n_neurons - n_neurons_threshold
    assert n_neurons_x > 0
    threshold = 0.8

    with nengo.Network(label="IA") as net:
        net.input = nengo.Node(size_in=d)
        x = nengo.networks.EnsembleArray(
            n_neurons_x, d,
            eval_points=nengo.dists.Uniform(0., 1.),
            intercepts=nengo.dists.Uniform(0., 1.),
            encoders=nengo.dists.Choice([[1.]]), label="Layer 1")
        net.x = x
        nengo.Connection(x.output, x.input, transform=tau_actual * A1 + I,
                         synapse=tau_actual)

        nengo.Connection(
            net.input, x.input,
            transform=tau_actual * B,
            synapse=tau_actual)

        with nengo.presets.ThresholdingEnsembles(0.):
            thresholding = nengo.networks.EnsembleArray(
                n_neurons_threshold, d, label="Layer 2")
            if share_thresholding_intercepts:
                for e in thresholding.ensembles:
                    e.intercepts = nengo.dists.Exponential(
                        0.15, 0., 1.).sample(n_neurons_threshold)
            net.output = thresholding.add_output('heaviside', lambda x: x > 0.)

        bias = nengo.Node(1., label="Bias")

        nengo.Connection(x.output, thresholding.input, synapse=0.005)
        nengo.Connection(
            bias, thresholding.input, transform=-threshold * np.ones((d, 1)))
        nengo.Connection(
            thresholding.heaviside, x.input,
            transform=tau_actual * A2, synapse=tau_actual)

    return net
