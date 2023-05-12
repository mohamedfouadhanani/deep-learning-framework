from dlf.callbacks.callback import Callback

class History(Callback):
    def __init__(self) -> None:
        super().__init__()
        
        """
        This callback should save:
            1. learning rate values
            2. training loss values
            3. validation loss if validation set was provided
        """
        
        self.cache = {
            "learning_rates": [],
            "training_losses": [],
            "validation_losses": []
        }
    
    def on_epoch_finish(self, trainer):
        # learning rate
        self.cache["learning_rates"].append(trainer.default_callbacks["optimizer"].learning_rate)
    