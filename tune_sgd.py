import torch
import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import (AverageMeterCollection, NUM_SAMPLES, BATCH_SIZE)

from ray.util.sgd.torch.constants import (
    SCHEDULER_STEP_EPOCH,
    NUM_STEPS,
    SCHEDULER_STEP_BATCH,    
)

from ray import tune
from ray.tune import CLIReporter
from ray.tune.utils.util import merge_dicts
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

from timeit import default_timer as timer
#import tqdm

from config import config as cfg
from training.run_management.run_manager import RunManager
from training.run_management.metrics.accuracy import Accuracy
from training.run_management.metrics.miou import ConfusionMatrix, ClassIoU, mIoU
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
# conf_mat = ConfusionMatrix(num_classes=3, from_logits=True, only_confmat=False)
# class_iou = ClassIoU()
# miou = mIoU()

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
    return (loss_pre + sum(loss_aux), loss_pre)


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
        #num_workers = 4 * len(cfg.cuda_devices)

        # Setup data loaders.
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=config[BATCH_SIZE], shuffle=True, num_workers=config["data_workers"],
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=config[BATCH_SIZE], shuffle=True, num_workers=config["data_workers"],
            pin_memory=True
        )
       
        # Setup model.
        model = BiSeNetV2(num_classes=55, output_aux=True)
        # if config["num_workers"] > 1:
        #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Setup optimizer.
        optimizer = build_optimizer(
            optimizer_type=cfg.optimizer_type,
            model=model,
            **{"lr": config["lr"], "momentum": config["momentum"], "weight_decay": config["weight_decay"]}
        )


        # Calculations for scheduler
        iter_per_epoch = len(train_set) / config[BATCH_SIZE] #/ config["num_workers"])
        max_iter = iter_per_epoch * config["max_num_epochs"]
        
        # Setup scheduler. - step size instead of max iter?
        scheduler = build_scheduler(
            scheduler_name=cfg.scheduler_name,
            optimizer = optimizer,
            **{"power": config["scheduler_power"], "max_iter": max_iter}
        )
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        # Setup loss.
        #criterion = OhemCrossEntropy2d(0.7)

        self.model, self.optimizer, self.scheduler = \
            self.register(
                models=model,
                optimizers=optimizer,
                schedulers=scheduler,
                # some of the possible args are shown at training startup
                apex_args={
                    'opt_level': "O1",
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
        (logits, *logits_aux) = self.model(images)

        # Loss calculation
        (loss, loss_pre) = criterion_train(logits, logits_aux, labels)

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
        loss_pre_value = loss_pre.item()
        acc_value = acc(logits, labels).item()
        # IoU calculation
        # (matrix, pred_count, true_count) = conf_mat(logits, labels)
        # class_iou_values = class_iou(matrix, pred_count, true_count)
        # mean_iou = miou(class_iou_values).item()
        lr = self.optimizer.param_groups[0]["lr"]

        return {"loss": loss_value, "loss_pre": loss_pre_value, "acc": acc_value, "lr": lr, "num_samples": len(images)} #"mIoU": mean_iou, 
    

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
                metric_meters.update(metrics, metrics.pop(NUM_SAMPLES, 1))

                if (batch_idx + 1) % 100 == 0:
                    print( metric_meters.summary() )
        
        return metric_meters.summary()

    
    def validate_batch(self, batch, batch_info):
        """
        Mini-batch iteration for the validation loop.
        """
        (images, labels) = batch
        (images, labels) = images.to(cfg.device), torch.squeeze(labels, 1).to(cfg.device)

        (logits, *logits_aux) = self.model(images)

        loss = criterion_val(logits, labels)

        # Logging
        loss_value = loss.item() 
        acc_value = acc(logits, labels).item()
        # IoU calculation
        # (matrix, pred_count, true_count) = conf_mat(logits, labels)
        # class_iou_values = class_iou(matrix, pred_count, true_count)
        # mean_iou = miou(class_iou_values).item()

        return {"val_loss": loss_value, "val_acc": acc_value}#, "val_mIoU": mean_iou}



def main(
    num_workers=3, use_gpu=True, num_cpus_per_worker=4, 
    max_num_epochs=75, use_fp16=False
    ):

    def step(trainer, info: dict):
        """
        Custom Tune training loop.
        """
        # https://docs.ray.io/en/master/raysgd/raysgd_tune.html#custom-training-step
        start_time = timer()
        train_metrics = trainer.train(profile=False)
        end_time = timer()
        print("Train epoch time: {} s".format(end_time - start_time))
        validation_metrics = trainer.validate(profile=False)
        # trainer.update_scheduler(metric=validation_metrics["val_loss"])
        all_stats = merge_dicts(train_metrics, validation_metrics)       
        return all_stats


    # start ray cluster (connect: ray.init(address="auto"))
    ray.init()

    print("CUDA Availability: {}".format(torch.cuda.is_available()))

    # instantiate metrics
    acc = Accuracy()

    # two configs setup: https://docs.ray.io/en/master/raysgd/raysgd_tune.html

    config = {
        "max_num_epochs": max_num_epochs,
        BATCH_SIZE : 16 * num_workers,
        "num_workers": num_workers,
        "data_workers": num_cpus_per_worker,
    }

    search_space = {
        "lr": tune.uniform(0.0001, 0.95), 
        "momentum": tune.uniform(0.1, 0.99),
        "weight_decay": tune.uniform(0.0001, 0.0009),
        "scheduler_power": tune.uniform(0.1, 0.99),
    }


    previous_hparams = [
        {
        "lr": 0.289526,
        "momentum": 0.86702,
        "weight_decay": 0.0001,
        "scheduler_power": 0.842195,
        },
        {
        "lr": 0.28441,
        "momentum": 0.873355,
        "weight_decay": 0.0001,
        "scheduler_power": 0.830296,
        },
        {
        "lr": 0.289642,
        "momentum": 0.866876,
        "weight_decay": 0.0001,
        "scheduler_power": 0.842508,
        },
    ]
    

    search_alg = BayesOptSearch(
        metric="loss",
        mode="min",
        random_search_steps=5,
        #points_to_evaluate = previous_hparams,
        verbose=3,
    )

    scheduler = ASHAScheduler(
        max_t=5, #5
        grace_period=2, #1
        reduction_factor=2 #2
    )

    trainer = TorchTrainer.as_trainable(
        #override_tune_step=step,
        training_operator_cls=MyTrainingOperator,
        scheduler_step_freq="batch",
        config=config,
        num_workers=num_workers,
        num_cpus_per_worker = num_cpus_per_worker,
        use_gpu=use_gpu,
        use_fp16=use_fp16,
        #wrap_ddp=True,
        add_dist_sampler=True,
        use_tqdm=True,
        backend="gloo"
    )

    # tune reporter setup
    reporter = CLIReporter(max_progress_rows=10)
    reporter.add_metric_column("loss")
    reporter.add_metric_column("val_acc")
    #reporter.add_metric_column("val_mIoU")
    

    analysis = tune.run(
        trainer,
        num_samples=25,
        config=search_space,
        metric="loss",
        mode="min",
        #stop={"training_iteration": 5},
        verbose=3,        
        search_alg=search_alg,
        scheduler=scheduler,
        local_dir="/home/rtx/ray_results",
        name = "my_06_17_experiment_a2d2",
        progress_reporter=reporter,
        #trial_name_creator = trial_name_string,
        #restore=cfg.ckpt_dir,
        #resume=True
    )

    best_trial = analysis.get_best_trial("loss", "min", "last")

    print("Best trial config: {}".format(best_trial.config))

    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"])
    )
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["val_acc"])
    )


    # CHECKPOINTING -restore experiment to continue
    # + RESTORE training to continue (main_sgd)
    #     if (epoch + 1) % 5 == 0:
    #         trainer.save('./my_ckpt.pth')

    # print("[INFO] Run Finished.")
    # trainer.save('./my_trained_model.pth')
    # trainer.shutdown()


if __name__ == '__main__':
    main()