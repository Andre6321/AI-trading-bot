class BaseStrategy:
    def __init__(self):
        pass

    def execute(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def set_parameters(self, **kwargs):
        pass

    def get_parameters(self):
        return {}