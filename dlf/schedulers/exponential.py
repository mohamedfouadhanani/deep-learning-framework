from dlf.callbacks.callback import Callback

class ExponentialScheduler(Callback):
    def __init__(self, decay_rate: float) -> None:
        super().__init__()

        self.iteration = 0
        self.decay_rate = decay_rate
    
    def on_epoch_finish(self, trainer):
        trainer.default_callbacks["optimizer"].learning_rate *= self.decay_rate ** (self.iteration + 1)
        self.iteration += 1