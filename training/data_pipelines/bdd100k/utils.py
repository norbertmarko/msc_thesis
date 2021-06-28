import torch
import numpy as np


class ToIntTensor(object):
    def __call__(self, image):  
        return torch.unsqueeze(torch.as_tensor(np.asarray(image), dtype=torch.int64), dim=0)


class ConvertToIntLabels_dd(object):
    def __init__(self):

        self.classes = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,
            (0, 0, 255): 2
        }

    def __call__(self, image):
        int_array = np.zeros(shape=(256, 512), dtype=int)
        image = np.asarray(image)

        # rgb to integer
        for rgb, idx in self.classes.items():
            int_array[(image==rgb).all(2)] = idx
        return int_array