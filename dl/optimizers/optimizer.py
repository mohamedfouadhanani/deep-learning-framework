class Optimizer:
    def __init__(self, learning_rate, batch_size, lr_decay=lambda lr0: lr0):
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        

    def __call__(self, layer):
        pass