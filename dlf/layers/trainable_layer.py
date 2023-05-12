from dlf.layers.layer import Layer
from dlf.trainable import Trainable

from dlf.initializers.initializer import Initializer

class TrainableLayer(Layer, Trainable):
    def __init__(self, params=None, initializer: Initializer=None) -> None:
        Layer.__init__(self)
        Trainable.__init__(self, params, initializer)