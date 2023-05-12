from dlf.layers.trainable_layer import TrainableLayer
from dlf.trainable import Trainable
from dlf.callbacks.callback import Callback

class Optimizer(Callback):

    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def update(self, layer: TrainableLayer):
        pass
    
    def on_params_update_start(self, trainer):
        for layer in trainer.model.layers:
            if not isinstance(layer, Trainable):
                continue
            
            if not layer.is_trainable:
                continue
            
            trainer.default_callbacks["optimizer"].update(layer)