from dlf.function import Function

class Loss(Function):
    def __init__(self, f, df) -> None:
        super().__init__()
        self.f = f
        self.df = df
    
    def forward(self, predictions, actuals):
        return self.f(predictions, actuals)
    
    def backward(self, predictions, actuals):
        return self.df(predictions, actuals)