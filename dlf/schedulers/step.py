from dlf.callbacks.callback import Callback

class StepScheduler(Callback):
    def __init__(self, step_size: int, decay_rate: float) -> None:
        super().__init__()
        
        self.step_size = step_size
        self.decay_rate = decay_rate
        
        self.count = 0
    
    def on_epoch_finish(self, trainer):
        self.count += 1
        
        if self.count == self.step_size:
            self.count = 0
            
            trainer.default_callbacks["optimizer"].learning_rate *= self.decay_rate