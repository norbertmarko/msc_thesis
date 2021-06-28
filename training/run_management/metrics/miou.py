import torch


class ConfusionMatrix:
    """
    Docstring.
    """
    def __init__(
            self,
            num_classes: int = 3,
            from_logits: bool = True,
            only_confmat: bool = True
        ):
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.only_confmat = only_confmat

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            y_pred, y_true = self._conversion(y_pred, y_true)

        # reshape tensors to 1D format (easier calculation)
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        # count the frequency of each value (class frequency here) 
        # in an array of non-negative ints.
        y_pred_count = torch.bincount(y_pred, weights=None, minlength=self.num_classes)
        y_true_count = torch.bincount(y_true, weights=None, minlength=self.num_classes)

        # calculate the category matrix 
        # (confusion matrix can be constructed from it)
        category_matrix = self.num_classes * y_true + y_pred

        # frequency count of category_matrix returns the confusion matrix in 1D format
        confusion_matrix_1d = torch.bincount(
            category_matrix, weights=None, minlength=self.num_classes ** 2
        )

        # final confusion matrix
        confusion_matrix = confusion_matrix_1d.reshape(self.num_classes, self.num_classes)

        if self.only_confmat:
            return confusion_matrix
        else:
            return (confusion_matrix, y_pred_count, y_true_count)

    def _conversion(self, y_pred, y_true):
        if len(y_pred.shape) == len(y_true.shape) + 1:
            y_pred = torch.argmax(y_pred, dim=1)
        return y_pred, y_true


class ClassIoU:
    """
    Docstring.
    """
    def __init__(self):
        pass

    def __call__(
            self,
            confmat: torch.Tensor,
            y_pred_count: torch.Tensor,
            y_true_count: torch.Tensor
        ) -> torch.Tensor:
        # A ⋂ B
        intersection = torch.diag(confmat)
        # A ⋃ B 
        union = y_pred_count + y_true_count - intersection
        # (A ⋂ B)/(A ⋃ B)
        class_iou = intersection / union 

        return class_iou


class mIoU:
    """
    Docstring.
    """
    def __init__(self):
        pass
    
    def __call__(self, class_iou: torch.Tensor) -> torch.Tensor:
        return self._nan_mean(class_iou)

    def _nan_mean(self, v, *args, inplace=False, **kwargs):
        """
        PyTorch implementation of np.nanmean
        to handle 0/0 cases due to absence of a class.
        # Source: https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
        """
        if not inplace:
            v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)