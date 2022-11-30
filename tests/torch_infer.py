from typing import List
import numpy as np
import torch

class TORCHInfer:
    def __init__(self, model_file) -> None:
        
        self.model = torch.jit.load(model_file)
        if torch.cuda.is_available():
            model = model.cuda()
    

    def predict(self, datas:List[torch.Tensor]):
        outputs = self.model(datas)
        return outputs