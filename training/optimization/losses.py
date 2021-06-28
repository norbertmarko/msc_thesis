import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCrossEntropy2d(nn.Module):

    """
    Non-weighted variant.

    OHEM, or Online Hard Example Mining, 
    is a bootstrapping technique that modifies SGD 
    to sample from examples in a non-uniform way 
    depending on the current loss of each example 
    under consideration.
    """

    def __init__(self, threshold, ignore_idx=19):
        super(OhemCrossEntropy2d, self).__init__()
        
        self.threshold = torch.log(
                            torch.tensor(
                                threshold, requires_grad=False, dtype=torch.float
                            )
                         ).cuda()
        self.ignore_idx = ignore_idx
        self.criteria = nn.CrossEntropyLoss(
                            ignore_index=ignore_idx, reduction='none'
                        )
        
    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_idx].numel()
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.threshold]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)
