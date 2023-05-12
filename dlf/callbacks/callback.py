from abc import ABC

class Callback(ABC):
    class EVENTS:
        ON_TRAINING_START = "on_training_start"
        ON_TRAINING_FINISH = "on_training_finish"
        
        ON_EPOCH_START = "on_epoch_start"
        ON_EPOCH_FINISH = "on_epoch_finish"
        
        ON_BATCH_START = "on_batch_start"
        ON_BATCH_FINISH = "on_batch_finish"
        
        ON_PARAMS_UPDATE_START = "on_params_update_start"
        ON_PARAMS_UPDATE_FINISH = "on_params_update_finish"
    
    def __init__(self) -> None:
        pass
    
    def on_training_start(self, trainer):
        pass
    
    def on_training_finish(self, trainer):
        pass
    
    def on_epoch_start(self, trainer):
        pass
    
    def on_epoch_finish(self, trainer):
        pass
    
    def on_batch_start(self, trainer):
        pass
    
    def on_batch_finish(self, trainer):
        pass
    
    def on_params_update_start(self, trainer):
        pass
    
    def on_params_update_finish(self, trainer):
        pass