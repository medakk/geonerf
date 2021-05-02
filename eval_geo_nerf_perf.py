import argparse
import glob
import os
import time
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf, sample_geodesics)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)


    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    assert(cfg.models.coarse.use_viewdirs == False)
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    # If a fine-resolution model is specified, initialize it.
    model_coarse.to(device)
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint, map_location=device)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    geo_nerf_model = models.GeoNeRF(128)
    geo_nerf_model.set_nerf(model_coarse)
    geo_nerf_model.to(device)
    state = torch.load(cfg.geonerf.out_model, map_location=device)
    geo_nerf_model.load_state_dict(state)
    geo_nerf_model.eval()

    geodesics = pickle.load(open(cfg.geonerf.cache_filename, 'rb'))
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {count_parameters(geo_nerf_model)}')

    # Initialize optimizer.
    N = geodesics['V'].shape[0]
    print(f"Perf evaluation on: {device}")
    X1 = geodesics['V'].copy().astype(np.float32)
    X1 = torch.from_numpy(X1).to(device)
    X1 = encode_position_fn(X1)
    n_trials = 100
    with torch.no_grad():
        start_time = 0.0
        for i in range(n_trials + 1):
            if i == 1:
                start_time = time.time() # first inference is always very slow, messes up bench
            x0 = geodesics['V'][random.randrange(0,50)].reshape((1, 3)).astype(np.float32)
            X0 = np.repeat(x0, N, axis=0)
            X0 = torch.from_numpy(X0).to(device)
            X0 = encode_position_fn(X0)
            pred = geo_nerf_model(X0, X1).cpu().numpy().reshape((-1,))
        end_time = time.time()
    avg_time = (end_time - start_time) / n_trials
    print(f'Average time for geodesics from a given source: {avg_time}')

    with torch.no_grad():
        start_time = time.time()
        for i in range(n_trials):
            x0 = geodesics['V'][random.randrange(0,N)].reshape((1, 3)).astype(np.float32)
            x1 = geodesics['V'][random.randrange(0,N)].reshape((1, 3)).astype(np.float32)

            X0 = encode_position_fn(torch.from_numpy(x0).to(device))
            X1 = encode_position_fn(torch.from_numpy(x1).to(device))
            pred = geo_nerf_model(X0, X1).cpu().numpy().reshape((-1,))
        end_time = time.time()
    avg_time = (end_time - start_time) / n_trials
    print(f'Average time for geodesics for a given pair: {avg_time}')
    

def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    main()
