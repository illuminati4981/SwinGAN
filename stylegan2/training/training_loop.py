# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from degradation.utils.common import instantiate_from_config
from torchvision.transforms import ToTensor
from metrics.fid import calculate_fid
from metrics.restoration_metrics import calculate_psnr_pt
from metrics.restoration_metrics import LPIPS
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# ----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [
                indices[(i + gw) % len(indices)] for i in range(len(indices))
            ]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)


# ----------------------------------------------------------------------------


def training_loop(
    swin,
    run_dir=".",  # Output directory.
    training_set_kwargs={},  # Options for training set.
    validation_set_kwargs={},  # Options for validation set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    G_kwargs={},  # Options for generator network.
    D_kwargs={},  # Options for discriminator network.
    G_opt_kwargs={},  # Options for generator optimizer.
    D_opt_kwargs={},  # Options for discriminator optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
    loss_kwargs={},  # Options for loss function.
    metrics=[],  # Metrics to evaluate during training.
    random_seed=0,  # Global random seed.
    num_gpus=1,  # Number of GPUs participating in the training.
    rank=0,  # Rank of the current process in [0, num_gpus[.
    batch_size=16,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu=4,  # Number of samples processed at a time by one GPU.
    ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup=None,  # EMA ramp-up coefficient.
    G_reg_interval=4,  # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
    augment_p=0,  # Initial value of augmentation probability.
    ada_target=None,  # ADA target value. None = fixed p.
    ada_interval=4,  # How often to perform ADA adjustment?
    ada_kimg=500,  # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg=25000,  # Total length of the training, measured in thousands of real images.
    kimg_per_tick=4,  # Progress snapshot interval.
    image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
    network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
    resume_pkl=None, # Network pickle to resume training from.
    resume_swin=None,  # Resume swin 
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    allow_tf32=False,  # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn=None,  # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
    train_config=None, # deg train config dict, config driiing
    val_config=None # deg val config dict, config drilling
):
    # Initialize.
    image_snapshot_ticks=500
    network_snapshot_ticks=500
    total_img = training_set_kwargs.max_size
    total_kimg = 48981
    start_time = time.time()
    device = torch.device("cuda", rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = (
        allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    )
    torch.backends.cudnn.allow_tf32 = (
        allow_tf32  # Allow PyTorch to internally use tf32 for convolutions
    )
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.


    # Degradation, Huggingface imagenet-1k and dataloader
    deg_train_transform = instantiate_from_config(train_config["batch_transform"])
    # deg_val_transform = instantiate_from_config(val_config["batch_transform"])


    # Load training set.
    if rank == 0:
        print("Loading training set...")
        
    training_set = dnnlib.util.construct_class_by_name(
        **training_set_kwargs
    )  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed
    )
    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set,
            sampler=training_set_sampler,
            batch_size=batch_size // num_gpus,
            **data_loader_kwargs,
        )
    )

    # Load validation set.
    if rank == 0:
        print("Loading validation set...")

    validation_set = dnnlib.util.construct_class_by_name(
        **validation_set_kwargs
    )  # subclass of training.dataset.Dataset
    validation_set_sampler = misc.InfiniteSampler(
        dataset=validation_set, rank=rank, num_replicas=num_gpus, seed=random_seed
    )
    validation_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=validation_set,
            sampler=validation_set_sampler,
            batch_size=batch_size // num_gpus,
            **data_loader_kwargs,
        )
    )

    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    if rank == 0:
        print()
        print("Num images: ", len(training_set))
        print("Image shape:", training_set.image_shape)
        print()

    # Construct networks.
    if rank == 0:
        print("Constructing networks...")
    common_kwargs = dict(
        c_dim=training_set.label_dim, 
        img_resolution=training_set.resolution,
        img_channels=training_set.num_channels,
    )

    G = (
        dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs)
        .train()
        .requires_grad_(False)
        .to(device)
    )  # subclass of torch.nn.Module
    D = (
        dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs)
        .train()
        .requires_grad_(False)
        .to(device)
    )  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        # for name, module in [("swin", swin), ("G", G), ("D", D), ("G_ema", G_ema)]:
        for name, module in [("G", G), ("D", D), ("G_ema", G_ema)]:
        # for name, module in [("swin", swin), ("G", G), ("G_ema", G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    
    if (resume_swin is not None) and (rank == 0):
        print(f'Resuming from "{resume_swin}"')
        with dnnlib.util.open_url(resume_swin) as f:
            resume_data = legacy.load_network_pkl(f)
        misc.copy_params_and_buffers(resume_data["swin"], swin, require_all=False)


    ### TODO: Fix Network Printing Problem or Ditch it
    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)


    # Setup augmentation.
    if rank == 0:
        print("Setting up augmentation...")
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = (
            dnnlib.util.construct_class_by_name(**augment_kwargs)
            .train()
            .requires_grad_(False)
            .to(device)
        )  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex="Loss/signs/real")

    # Distribute across GPUs.
    if rank == 0:
        print(f"Distributing across {num_gpus} GPUs...")
    ddp_modules = dict()
    for name, module in [
        ("swin", swin),
        ("G_mapping", G.mapping),
        ("G_synthesis", G.synthesis),
        ("D", D),
        (None, G_ema),
        ("augment_pipe", augment_pipe),
    ]:
        if (
            (num_gpus > 1)
            and (module is not None)
            and len(list(module.parameters())) != 0
        ):
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(
                module, device_ids=[device], broadcast_buffers=False
            )
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print("Setting up training phases...")
    loss = dnnlib.util.construct_class_by_name(
        device=device, **ddp_modules, **loss_kwargs
    )  # subclass of training.loss.Loss
    phases = []


    ### Swin Optimizer Initialization
    if rank == 0:
        swin_start_event = torch.cuda.Event(enable_timing=True)
        swin_end_event = torch.cuda.Event(enable_timing=True)
    swin_opt = dnnlib.util.construct_class_by_name(
        params=swin.parameters(), **G_opt_kwargs
    )

    for name, module, opt_kwargs, reg_interval in [
        ("G", G, G_opt_kwargs, G_reg_interval),
        ("D", D, D_opt_kwargs, D_reg_interval),
    ]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(
                params=module.parameters(), **opt_kwargs
            )  # subclass of torch.optim.Optimizer
            phases += [
                dnnlib.EasyDict(name=name + "both", module=module, opt=opt, interval=1)
            ]
        else:  # .
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta**mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(
                module.parameters(), **opt_kwargs
            )  # subclass of torch.optim.Optimizer
            phases += [
                dnnlib.EasyDict(name=name + "main", module=module, opt=opt, interval=1)
            ]
            phases += [
                dnnlib.EasyDict(
                    name=name + "reg", module=module, opt=opt, interval=reg_interval
                )
            ]


    ### Testing: LPIPS + L1
    # G_phase = None
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)
        if phase.name == 'Gmain':
          G_phase = phase


    # Export sample images.
    grid_z = None
    grid_c = None
    if rank == 0:
        print("Exporting sample images...")
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print("Initializing logs...")
    stats_collector = training_stats.Collector(regex=".*")
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "wt")
        try:
            import torch.utils.tensorboard as tensorboard

            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print("Skipping tfevents export:", err)

    # Train.
    if rank == 0:
        print(f"Training for {total_kimg} kimg...")
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    cur_time = tick_start_time - start_time
    batch_idx = 0
    losses, fids, psnrs, lpips = [], [], [], []
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    
    # Training Loop
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function("data_fetch"):                
            print(f'Iteration: {cur_tick}')
            print(f'Batch: {round(cur_tick / total_img, 1)}')
            print(f'Handled Images: {cur_nimg}/{total_kimg}')
            phase_gt_img, phase_real_c = next(training_set_iterator) 
            phase_gt_img = torch.clamp((phase_gt_img).round(), 0, 255) / 255. # FIXME temperature fix
            
            transformed_result = deg_train_transform(phase_gt_img.to(torch.float32)) # degrade the batch of imgs as function
            # (batch_size, C, H, W)
            phase_real_img = transformed_result["jpg"].permute(0, 3, 1, 2)
            phase_deg_img = transformed_result["hint"].permute(0, 3, 1, 2)

            phase_real_img = (
                phase_real_img.to(device).to(torch.float32) / 127.5 - 1
            ).split(batch_gpu)
            phase_deg_img = (
                phase_deg_img.to(device).to(torch.float32) / 127.5 - 1
            ).split(batch_gpu)

            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [
                phase_gen_z.split(batch_gpu)
                for phase_gen_z in all_gen_z.split(batch_size)
            ]
            all_gen_c = [
                # get a random img from the training_set and get its label
                training_set.get_label(np.random.randint(len(training_set)))
                for _ in range(len(phases) * batch_size)
            ]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [
                phase_gen_c.split(batch_gpu)
                for phase_gen_c in all_gen_c.split(batch_size)
            ]

        # Execute training phases.
        print('Execute training phases')
        loss_values = []


        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            if (phase.name == 'Gmain' or phase.name == 'Greg'):
              swin_opt.zero_grad(set_to_none=True)
              swin.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, deg_img, real_c, gen_z, gen_c) in enumerate(
                zip(phase_real_img, phase_deg_img, phase_real_c, phase_gen_z, phase_gen_c)
            ):
                sync = round_idx == batch_size // (batch_gpu * num_gpus) - 1
                gain = phase.interval
                loss_value = loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    deg_img = deg_img,
                    real_c=real_c,
                    gen_z=gen_z,
                    gen_c=gen_c,
                    sync=sync,
                    gain=gain,
                )

                if (phase.name == 'Gmain'):
                  loss_values.append(loss_value.item())

            # Update weights.
            phase.module.requires_grad_(False)
            if (phase.name == 'Gmain' or phase.name == 'Greg' or phase.name == 'Dmain'):
                swin.requires_grad_(False)
                
            with torch.autograd.profiler.record_function(phase.name + "_opt"):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(
                            param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                        )
                phase.opt.step()

                # Update weights of swin transformer
                if (phase.name == 'Gmain' or phase.name == 'Greg' or phase.name == 'Dmain'):      
                  for param in swin.parameters():
                      if param.grad is not None:
                          misc.nan_to_num(
                              param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                          )
                  swin_opt.step()

            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))


        # Update G_ema.
        print('Update G_ema')
        with torch.autograd.profiler.record_function("Gema"):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = (
                np.sign(ada_stats["Loss/signs/real"] - ada_target)
                * (batch_size * ada_interval)
                / (ada_kimg * 1000)
            )
            augment_pipe.p.copy_(
                (augment_pipe.p + adjust).max(misc.constant(0, device=device))
            )


        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()

        # Check for abort.
        done = (cur_nimg >= total_kimg)
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print("Aborting...")
                print("--- Statistics ---")
                print('fids: ', fids)
                print('psnrs: ', psnrs)
                print('lpips: ', lpips)


        # Validation and Image Saving.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
          print('Validation...')
          val_img, phase_real_c = next(validation_set_iterator) 
          val_img = torch.clamp((val_img).round(), 0, 255) / 255.
          
          transformed_result = deg_train_transform(val_img.to(torch.float32))
          val_real_img = transformed_result["jpg"].permute(0, 3, 1, 2)
          val_deg_img = transformed_result["hint"].permute(0, 3, 1, 2)

          # Normalization
          val_real_img = val_real_img.to('cpu').to(torch.float32) / 127.5 - 1
          val_deg_img = val_deg_img.to('cpu').to(torch.float32) / 127.5 - 1

          swin_input = val_deg_img.to('cuda')    
          gen_img, size128_output, size64_output, size32_output, size16_output, size8_output, size4_output = swin(swin_input)
          noises = [size128_output, size64_output, size32_output, size16_output, size8_output, size4_output]

          G = G.to('cpu')
          G_ema = G_ema.to('cuda')

          gen_img = G_ema.mapping(gen_img, grid_c[0])
          gen_img = G_ema.synthesis(gen_img, noises).cpu()
          gen_img = gen_img.to('cpu')
          deg_img = val_deg_img.to('cpu')

          print('Saving Image Snapshot...')
          save_img = torch.cat([val_real_img, deg_img, gen_img]).numpy()
          save_image_grid(save_img, os.path.join(run_dir, f'result_{batch_idx}.png'), drange=[-1,1], grid_size=(batch_size, 3))

          G = G.to('cuda')
          G_ema = G_ema.to('cuda')

          # # Evaluate metrics.
          print('Evaluating Model...')
          losses.append(loss_values)
          print('Training losses: ', losses)

          # FID
          # Better closed to 0
          fid = calculate_fid(batch_size, val_real_img, gen_img)
          fids.append(fid)
          print('fids: ', fids)

          # # PSNR
          # For current implementation, when it is closed to 80, it is better.
          psnr = calculate_psnr_pt(val_real_img, gen_img, 0, batch_size)
          psnrs.append(psnr.item())
          print('psnrs: ', psnrs)

          # LPIPS
          # Better closed to 0
          lpips_model = LPIPS(net='alex').to('cpu')
          lpips_value = lpips_model(val_real_img, gen_img, False, batch_size)
          lpips.append(lpips_value.item())
          print('lpips: ', lpips)


        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (
            done or cur_tick % network_snapshot_ticks == 0
        ):
          print('Saving Network Snapshot...')
          snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs)) 
          for name, module in [
              ("swin", swin), ### Swin Transformer Saving
              ("G", G),
              ("D", D),
              ("G_ema", G_ema),
              ("augment_pipe", augment_pipe),
          ]:
              if module is not None:
                  if num_gpus > 1:
                      misc.check_ddp_consistency(module, ignore_regex=r".*\.w_avg")
                  module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
              snapshot_data[name] = module
              del module  # conserve memory
          snapshot_pkl = os.path.join(
              run_dir, f"network-snapshot-{batch_idx:06d}.pkl"
          )
          if rank == 0:
              with open(snapshot_pkl, "wb") as f:
                  pickle.dump(snapshot_data, f)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        cur_time = tick_start_time - tick_end_time
        if done:
            break
        print()

    # Done.
    if rank == 0:
        print()
        print("Exiting...")


# ----------------------------------------------------------------------------
