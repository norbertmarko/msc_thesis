import torch
import torch.nn as nn


def build_optimizer(optimizer_type:str, model:nn.Module, **kwargs):

    if optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), **kwargs)
    
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)

    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), **kwargs)
    
    raise NotImplementedError(optimizer_type)