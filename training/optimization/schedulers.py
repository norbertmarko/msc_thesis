import torch

# SEGMENTATION

class WarmupLrScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
            self,
            optimizer,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            ratio = self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


class WarmupPolyLrScheduler(WarmupLrScheduler):

    def __init__(
            self,
            optimizer,
            power,
            max_iter,
            warmup_iter=1000,
            warmup_ratio=0.1,
            warmup='exp',
            last_epoch=-1,
    ):
        self.power = power
        self.max_iter = max_iter
        super(WarmupPolyLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio

# OBJ DETECTION
class PolyLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, gamma=0.9, n_iteration=-1):
        self.step_size = n_iteration
        self.gamma = gamma
        super(PolyLR, self).__init__(optimizer)

    def get_lr(self):
        decay = 1 - self._step_count / float(self.step_size)
        decay = max(0., decay) ** self.gamma
        return [base_lr * decay for base_lr in self.base_lrs]


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    
    From:
        https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    """

    def __init__(self, optimizer, multiplier:float, total_epoch:int, after_scheduler_cfg=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = build_scheduler(after_scheduler_cfg, optimizer)
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def build_scheduler(scheduler_name:str, optimizer, **kwargs) \
    -> torch.optim.lr_scheduler._LRScheduler:

    if scheduler_name is None:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, 1.0)

    if scheduler_name.lower() == 'StepLR'.lower():
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)

    if scheduler_name.lower() == 'MultiStepLR'.lower():
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)

    if scheduler_name.lower() == 'ExponentialLR'.lower():
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)

    if scheduler_name.lower() == 'CosineAnnealingLR'.lower():
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    if scheduler_name.lower() == 'PolyLR'.lower():
        return PolyLR(optimizer, **kwargs)

    if scheduler_name.lower() == 'GradualWarmupScheduler'.lower():
        return GradualWarmupScheduler(optimizer, **kwargs)

    # go over
    if scheduler_name.lower() == 'WarmupLrScheduler'.lower():
        return WarmupLrScheduler(optimizer, **kwargs)

    if scheduler_name.lower() == 'WarmupPolyLrScheduler'.lower():
        return WarmupPolyLrScheduler(optimizer, **kwargs)
    
    raise NotImplementedError(scheduler_name)

