import argparse
import glob
import os
import time
import pickle

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

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    """
    # Disable Cache dir stuff
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
        print("using cache!")
    else:
    """
    if True:
        # Load dataset
        images, poses, render_poses, hwf = None, None, None, None
        assert(cfg.dataset.type.lower() == "blender")
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])

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
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    geodesics = pickle.load(open(cfg.geonerf.cache_filename, 'rb'))
    
    geo_nerf_model = models.GeoNeRF(128)
    geo_nerf_model.set_nerf(model_coarse)
    geo_nerf_model.to(device)

    # Initialize optimizer.
    trainable_parameters = list(geo_nerf_model.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=0.0001)

    
    n_iters = 100000
    batch_size = 64
    # encode_position_fn = lambda x: x
    for i in trange(n_iters):
        optimizer.zero_grad()
        geo_nerf_model.train()

        X0, X1, targets = sample_geodesics(geodesics, batch_size, 'train')
        X0, X1, targets = X0.to(device), X1.to(device), targets.to(device)
        X0, X1 = encode_position_fn(X0), encode_position_fn(X1)

        pred = torch.flatten(geo_nerf_model(X0, X1))

        loss = torch.nn.functional.mse_loss(pred, targets)
        loss.backward()
        optimizer.step()

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
            )

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            geo_nerf_model.eval()

            start = time.time()
            with torch.no_grad():
                X0, X1, targets = sample_geodesics(geodesics, batch_size, 'val')
                X0, X1, targets = X0.to(device), X1.to(device), targets.to(device)
                X0, X1 = encode_position_fn(X0), encode_position_fn(X1)
                pred = torch.flatten(geo_nerf_model(X0, X1))
                loss = torch.nn.functional.mse_loss(pred, targets)

                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Time: "
                    + str(time.time() - start)
                )

        if False and i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")

    geo_nerf_model.eval()
    N = geodesics['V'].shape[0]
    D = np.ndarray((N, N))
    print("Final evaluation")
    X1 = geodesics['V'].copy().astype(np.float32)
    X1 = torch.from_numpy(X1).to(device)
    X1 = encode_position_fn(X1)
    with torch.no_grad():
        for i in trange(N):
            x0 = geodesics['V'][i].reshape((1, 3)).astype(np.float32)
            X0 = np.repeat(x0, N, axis=0)
            X0 = torch.from_numpy(X0).to(device)
            X0 = encode_position_fn(X0)
            pred = geo_nerf_model(X0, X1).cpu().numpy().reshape((-1,))
            D[i] = pred
    np.save(cfg.geonerf.out_filename, D)
    torch.save(geo_nerf_model.state_dict(), cfg.geonerf.out_model)
    plt.imshow(D); plt.show()



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
