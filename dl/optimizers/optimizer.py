class Optimizer:
    def __init__(self, learning_rate, lr_decay=lambda lr0: lr0):
        self.learning_rate0 = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        

    def __call__(self, layer):
        pass