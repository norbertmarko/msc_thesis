import torch
import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import (AverageMeterCollection, NUM_SAMPLES)

from ray.util.sgd.torch.constants import (
    SCHEDULER_STEP_EPOCH,
    NUM_STEPS,
    SCHEDULER_STEP_BATCH,    
)

from ray import tune

from timeit import default_timer as timer
#import tqdm

from config import config as cfg
from training.run_management.run_manager import RunManager
from training.run_management.metrics.accuracy import Accuracy
from training.optimization.losses import OhemCrossEntropy2d
from training.optimization.optimizers import build_optimizer
from training.optimization.schedulers import build_scheduler

#from training.data_pipelines.bdd100k.data_loader import train_set, val_set
from training.data_pipelines.a2d2.data_loader import train_set, val_set
from training.models.headnets.bisenet_v2_head import BiSeNetV2

# Nvidia APEX
try:
    from apex import amp
except ImportError:
    amp = None

#initialize metrics
acc = Accuracy()

# Custom losses
def criterion_train(logits, logits_aux, labels, threshold=0.7):
    """
    Custom loss calculation for the training loop.
    """
    criterion_main = OhemCrossEntropy2d(threshold)
    criterion_aux = [
        OhemCrossEntropy2d(threshold) for _ in range(cfg.num_aux_heads)
    ]

    loss_pre = criterion_main(logits, labels)
    loss_aux = [
        crit(lgts, labels) \
            for (crit, lgts) in zip(criterion_aux, logits_aux)
    ]
    return loss_pre + sum(loss_aux)


def criterion_val(logits, labels, threshold=0.7):
    """
    Custom loss calculation for the validation loop.
    """
    criterion_main = OhemCrossEntropy2d(threshold)

    return criterion_main(logits, labels)


# Training Operator
class MyTrainingOperator(TrainingOperator):
    def setup(self, config):

        # initial parameters (put into cfg later)
        num_workers = 4 * len(cfg.cuda_devices)

        # Setup data loaders.
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers
        )

        # Setup model.
        model = BiSeNetV2(num_classes=55, output_aux=False)
        # if config["num_workers"] > 1:
        #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Setup optimizer.
        optimizer = build_optimizer(
            optimizer_type=cfg.optimizer_type,
            model=model,
            **{"lr": config["lr"], "momentum": config["momentum"], "weight_decay": config["weight_decay"]}
        )


        # Calculations for scheduler
        iter_per_epoch = len(train_set) / config["batch_size"]
        max_iter = iter_per_epoch * cfg.max_epochs
        

        # # Setup scheduler. - step size instead of max iter?
        # scheduler = build_scheduler(
        #     scheduler_name=cfg.scheduler_name,
        #     optimizer = optimizer,
        #     **{"power": config["scheduler_power"], "max_iter": max_iter}
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        # Setup loss.
        criterion = OhemCrossEntropy2d(0.7)

        self.model, self.optimizer, self.criterion, self.scheduler = \
            self.register(
                models=model,
                optimizers=optimizer,
                criterion=criterion,
                schedulers=scheduler, 
                # some of the possible args are shown at training startup
                apex_args={
                    'opt_level': "O3",
                    'verbosity': 1,
                }
            )
        self.register_data(train_loader=train_loader, validation_loader=val_loader)


    def train_epoch(self, iterator, info):
        """
        Training loop override.
        """
        
        model = self.model
        scheduler = None
        if hasattr(self, "scheduler"):
            scheduler = self.scheduler

        metric_meters = AverageMeterCollection()

        # switch to training mode
        model.train()
        for (batch_idx, batch) in enumerate(iterator):
            batch_info = {
                "batch_idx": batch_idx,
                "global_step": self.global_step
            }
            batch_info.update(info)
            metrics = self.train_batch(batch, batch_info=batch_info)

            if scheduler and self.scheduler_step_freq == SCHEDULER_STEP_BATCH:
                scheduler.step()

            metric_meters.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))

            if (batch_idx + 1) % 100 == 0:
                print( metric_meters.summary() )

            self.global_step += 1

        if scheduler and self.scheduler_step_freq == SCHEDULER_STEP_EPOCH:
            scheduler.step()       

        return metric_meters.summary()


    def train_batch(self, batch, batch_info):
        """
        Mini-batch iteration for the training loop.
        """
        (images, labels) = batch
        (images, labels) = images.to(cfg.device), torch.squeeze(labels, 1).to(cfg.device)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward Pass
        #(logits, *logits_aux) = self.model(images)
        logits = self.model(images)

        # Loss calculation
        loss = self.criterion(logits, labels)

        # FP16 mixed precision training if enabled 
        if self.use_fp16 and amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        # is this needed?
        torch.cuda.synchronize()

        # Logging
        loss_value = loss.item() 
        acc_value = acc(logits, labels).item()
        lr = self.optimizer.param_groups[0]["lr"]

        return {"loss": loss_value, "acc": acc_value, "lr": lr, "num_samples": len(batch)}
    

    def vaildate(self, val_iterator, info):
        """
        Validation loop override.
        """

        model = self.model
        metric_meters = AverageMeterCollection()

        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(val_iterator):
                batch_info = {"batch_idx": batch_idx}
                batch_info.update(info)
                metrics = self.validate_batch(batch, batch_info)
                metric_meters.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))

                if (batch_idx + 1) % 100 == 0:
                    print( metric_meters.summary() )
        
        return metric_meters.summary()

    
    def validate_batch(self, batch, batch_info):
        """
        Mini-batch iteration for the validation loop.
        """
        (images, labels) = batch
        (images, labels) = images.to(cfg.device), torch.squeeze(labels, 1).to(cfg.device)

        #(logits, *logits_aux) = self.model(images)
        logits = self.model(images)

        loss = self.criterion(logits, labels)

        # Logging
        loss_value = loss.item() 
        acc_value = acc(logits, labels).item()

        return {"val_loss": loss_value, "val_acc": acc_value}



def main(num_workers=1, use_gpu=True, data_workers=8):

    # start ray cluster (connect: ray.init(address="auto"))
    ray.init()

    print("CUDA Availability: {}".format(torch.cuda.is_available()))

    # instantiate metrics
    acc = Accuracy()

    config = {
        "batch_size": 10,
        "lr": 0.0424016,
        "momentum": 0.85,
        "weight_decay": 0.0004,
        "scheduler_power": 0.9,
        "max_iter": 175000,
        "num_workers": num_workers,
        "data_workers": data_workers
    }

    trainer = TorchTrainer(
        training_operator_cls=MyTrainingOperator,
        scheduler_step_freq='epoch',
        config=config,
        num_workers=num_workers,
        add_dist_sampler=True,
        use_gpu=use_gpu,
        use_fp16=True,
        backend="nccl",
        use_tqdm=True,
    )

    # training the model
    for epoch in range(cfg.max_epochs):
        start_time = timer()
        train_metrics = trainer.train()
        end_time = timer()
        print("Train epoch time: {} s".format(end_time - start_time))
        print("Training epoch summary: {}".format(train_metrics))
        validation_metrics = trainer.validate()
        print("Validation epoch summary: {}".format(validation_metrics))
        
        if (epoch + 1) % 5 == 0:
            trainer.save('./my_ckpt.pth')

    print("[INFO] Run Finished.")
    trainer.save('./my_trained_model.pth')
    trainer.shutdown()


if __name__ == '__main__':
    main()