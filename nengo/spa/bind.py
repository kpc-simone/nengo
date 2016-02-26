import nengo
from nengo.exceptions import ValidationError
from nengo.spa.module import Module


class Bind(Module):
    """A module for binding together both inputs

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the two vectors to be compared
    vocab : Vocabulary, optional
        The vocabulary to use to interpret the vectors
    n_neurons : int
        Number of neurons to use in each product computation
    invert_a, invert_b : bool
        Whether to reverse the order of elements in either
        the first input (`invert_a`) or the second input (`invert_b`).
        Flipping the second input will make the network perform circular
        correlation instead of circular convolution.
    input_magnitude : float
        The expected magnitude (vector norm) of the two input values.

    label : str, optional
        A name for the ensemble. Used for debugging and visualization.
        Default: None
    seed : int, optional
        The seed used for random number generation.
        Default: None
    add_to_container : bool, optional
        Determines if this Network will be added to the current container.
        Defaults to true iff currently with a Network.
    """
    def __init__(self, dimensions, vocab=None, n_neurons=200, invert_a=False,
                 invert_b=False, input_magnitude=1.0, label=None, seed=None,
                 add_to_container=None):
        super(Bind, self).__init__(label, seed, add_to_container)
        if vocab is None:
            # use the default vocab for this number of dimensions
            vocab = dimensions
        elif vocab.dimensions != dimensions:
            raise ValidationError(
                "Dimensionality of given vocabulary (%d) does "
                "not match dimensionality of buffer (%d)" %
                (vocab.dimensions, dimensions), attr='dimensions', obj=self)

        with self:
            self.cc = nengo.networks.CircularConvolution(
                n_neurons, dimensions, invert_a, invert_b,
                input_magnitude=input_magnitude)
            self.A = self.cc.A
            self.B = self.cc.B
            self.output = self.cc.output

        self.inputs = dict(A=(self.A, vocab), B=(self.B, vocab))
        self.outputs = dict(default=(self.output, vocab))
