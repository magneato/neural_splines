class BaseNeuralModel:
    def __init__(self,*a,**k): pass
    @classmethod
    def from_pretrained(cls,*a,**k):
        raise NotImplementedError("Model loading not implemented.")
