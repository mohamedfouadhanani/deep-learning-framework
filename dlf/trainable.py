from abc import ABC

from dlf.initializers.initializer import Initializer
from dlf.initializers.random_normal import RandomNormal


class Trainable(ABC):

    def __init__(self, params, initializer: Initializer = None) -> None:
        super().__init__()

        self.params = params if params is not None else []
        self.is_trainable = True

        self.initializer = RandomNormal() if initializer is None else initializer
