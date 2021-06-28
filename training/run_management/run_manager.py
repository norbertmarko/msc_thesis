import time
import datetime
from collections import OrderedDict
import math

import pandas as pd

from config import config as cfg


class RunManager:
    """
    Docstring here.
    """
    def __init__(self):

        self.iter = 0
        self.epoch = 0

        self.loss_meter = LossMeter('loss')
        self.loss_pre_meter = LossMeter('loss_pre')
        self.loss_aux_meters = [LossMeter('loss_aux{}'.format(i)) for i in range(cfg.num_aux_heads)]
        self.loss_val_meter = LossMeter('loss_val') # Main metric passed to Ray Tune.

        self.acc_meter = AccMeter('acc')
        self.acc_val_meter = AccMeter('acc_val')

        self.run_start_time = None
        self.epoch_start_time = None

        self.model = None
        self.tb = None
        self.path = None
        

    def begin_run(self):
        self.run_start_time = time.time()


    def end_run(self):
        run_duration = time.time() - self.run_start_time
        

    def begin_epoch(self):

        self.loss_meter.reset()
        self.loss_pre_meter.reset()
        _ = [meter.reset() for meter in self.loss_aux_meters]
        self.loss_val_meter.reset()

        self.acc_meter.reset()
        self.acc_val_meter.reset()

        self.epoch_start_time = time.time()


    def end_epoch(self):
        
        epoch_duration = time.time() - self.epoch_start_time

        loss = self.loss_meter.mean()
        loss_pre = self.loss_pre_meter.mean()
        loss_aux_meters = [meter.mean() for meter in self.loss_aux_meters]
        loss_val = self.loss_val_meter.mean()

        acc = self.acc_meter.mean()
        acc_val = self.acc_val_meter.mean()

        self.epoch += 1

        print("Epoch duration: {} s".format(epoch_duration))

        # # Build Ordered Dictionary with run data
        # results = OrderedDict()
        # results["epoch"] = self.epoch
        # results["train loss"] = train_loss
        # results["train accuracy"] = train_accuracy
        # results["val loss"] = val_loss
        # results["val accuracy"] = val_accuracy
        # results["epoch duration"] = epoch_duration
        # results["run duration"] = run_duration
        
        # for (k, v) in self.run_params._asdict().items(): results[k] = v
        # self.run_data.append(results)
        # df = pd.DataFrame.from_dict(self.run_data, orient='columns')


    def save(self):
        """
        Saves the model, outputs Pandas data and plotted graphs.
        """
        pass


class LossMeter(object):
    def __init__(self, name):
        self.name = name
        self.running_loss = []

    def update(self, loss):
        self.running_loss.append( loss.item() )

    def mean(self):
        """
        Calculates mean loss. This becomes the epoch
        loss when called at the end of the epoch.
        """
        return sum(self.running_loss) / len(self.running_loss)

    def reset(self):
        self.running_loss = []


class AccMeter(object):
    def __init__(self, name):
        self.running_acc = []

    # include item() method so the two classes can be fused
    def update(self, acc):
        self.running_acc.append( acc ) 
    
    def mean(self):
        """
        Calculates mean acc. This becomes the epoch
        acc when called at the end of the epoch.
        """
        return sum(self.running_acc) / len(self.running_acc)

    def reset(self):
        self.running_acc = []