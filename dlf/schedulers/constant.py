from dlf.callbacks.callback import Callback

class ConstantScheduler(Callback):
    def __init__(self) -> None:
        super().__init__()