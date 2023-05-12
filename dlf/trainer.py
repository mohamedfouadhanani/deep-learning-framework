from typing import List, Dict

from dlf.sequential import Sequential
from dlf.losses.loss import Loss
from dlf.trainable import Trainable
from dlf.optimizers.optimizer import Optimizer
from dlf.callbacks.callback import Callback
from dlf.callbacks.history import History as HistoryCallback
from dlf.schedulers.constant import ConstantScheduler

class Trainer:
    
    def __init__(self, model: Sequential, optimizer: Optimizer, loss: Loss, lr_scheduler: Callback = None) -> None:
        self.model = model
        # self.optimizer = optimizer
        self.loss = loss
        
        # initializer callbacks list
        self.default_callbacks: Dict[str, Callback] = {
            "history": HistoryCallback(),
            "learning_rate": ConstantScheduler() if lr_scheduler is None else lr_scheduler,
            "optimizer": optimizer
        }
    
    def batch(self, examples_set, batch_size):           
        example_inputs, example_labels = examples_set
        
        examples_size = len(example_inputs)

        for index in range(0, examples_size, batch_size):
            start = index
            finish = min(start + batch_size, examples_size - 1)
            
            
            yield example_inputs[start:finish], example_labels[start:finish]
    
    def run_callbacks_of_event(self, callbacks: List[Callback], event: str):
        for callback in callbacks:
            fn = getattr(callback, event)
            fn(self)
    
    def fit(self, training_set, n_epochs, batch_size, callbacks: List[Callback] = [], **kwargs):
        # extending the callback list with the ones provided by the user
        callbacks.extend(self.default_callbacks.values())
        
        # start training callbacks
        self.run_callbacks_of_event(callbacks, Callback.EVENTS.ON_TRAINING_START)
        
        verbose = kwargs.get("verbose", True)
        
        for epoch in range(n_epochs):
            # start epoch callbacks
            self.run_callbacks_of_event(callbacks, Callback.EVENTS.ON_EPOCH_START)
            
            batches = self.batch(training_set, batch_size)
            for batch_inputs, batch_labels in batches:
                # start batch callbacks
                self.run_callbacks_of_event(callbacks, Callback.EVENTS.ON_BATCH_START)
                
                # forward propagation
                predictions = self.model.forward(batch_inputs)

                # dL / doutputs
                dpredictions = self.loss.backward(predictions, batch_labels)
                
                # backward propagations
                self.model.backward(dpredictions)
                
                # parameters update
                self.run_callbacks_of_event(callbacks, Callback.EVENTS.ON_PARAMS_UPDATE_START)
                self.run_callbacks_of_event(callbacks, Callback.EVENTS.ON_PARAMS_UPDATE_FINISH)
                
                # finish batch callbacks
                self.run_callbacks_of_event(callbacks, Callback.EVENTS.ON_BATCH_FINISH)

            # finish epoch callbacks
            self.run_callbacks_of_event(callbacks, Callback.EVENTS.ON_EPOCH_FINISH)
            
            training_inputs, training_labels = training_set
            l = self.loss.forward(self.model.forward(training_inputs), training_labels)
            
            if verbose:
                print(f"[{epoch + 1}/{n_epochs}]: Training Loss = {l}")
                pass
        

        # finish training callbacks
        self.run_callbacks_of_event(callbacks, Callback.EVENTS.ON_TRAINING_FINISH)
        
        return self.default_callbacks["history"].cache