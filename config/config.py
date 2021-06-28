num_aux_heads = 4
cuda_devices = [0, 1, 2]
device = "cuda"

max_epochs = 2

optimizer_type = 'sgd'
scheduler_name = 'WarmupPolyLrScheduler'

ckpt_dir = f'/home/rtx/markonorbert/markonorbert/MSc_thesis/src/my_raytune_results/my_first_experiment/WrappedDistributedTorchTrainable_0e501_00011_11_batch_size=8,lr=0.042402,max_iter=4.375e+04,momentum=0.85,scheduler_power=0.9,weig_2021-03-15_05-13-09/checkpoint_5'
