# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch.nn.functional as F
from wisp.utils import PerfTimer
from wisp.ops.differential import finitediff_gradient

import trimesh

import os


def trace_num_points(pipeline, num_points=2**20):

    surface_points = []

    current_num_points = 0

    while True:

        dict = trace(pipeline.nef)

        x = dict['x'][dict['hit']*dict['within_dist'][:, 0]]

        valid = pipeline.nef.grid.query(x)

        x = x[valid != -1]

        surface_points.append(x)

        current_num_points += x.shape[0]

        if(current_num_points >= num_points):
            break

    surface_points = torch.cat(surface_points, dim=0)
    surface_points = surface_points[:num_points]
    return surface_points

def create_num_dirs(num=1000):

    dirs = torch.randn(num, 3).cuda()

    dirs = F.normalize(dirs, dim=1)

    return dirs


def trace(nef, num_steps=64, min_dist=1e-3, step_size=1.0, max_dist=100, num_points=100_000):

    ray_o = torch.rand(num_points, 3).cuda()*2-1
    ray_d = create_num_dirs(num_points).cuda()

    while True:

        # print("ray_o.shape beginning")
        # print(ray_o.shape)
        
        create_num_dirs(max(num_points-ray_o.shape[0], 10000)).cuda()

        ray_o = torch.cat(
            [ray_o, create_num_dirs(max(num_points-ray_o.shape[0], 10000)).cuda()], dim=0)

        # print("ray_o.shape middle")
        # print(ray_o.shape)

        valid = pipeline.nef.grid.query(ray_o)
        # print("valid")
        # print(valid)

        ray_o = ray_o[valid != -1]

        # print("ray_o.shape end ", ray_o.shape)

        if(ray_o.shape[0] >= num_points):
            ray_o = ray_o[:num_points]
            break

    #
    # print("after while loop")

    """PyTorch implementation of sphere tracing."""
    timer = PerfTimer(activate=False)
    supported_channels = nef.get_supported_channels()
    assert "sdf" in supported_channels and "this tracer requires sdf channels"

    # Distanace from ray origin
    t = torch.zeros(ray_o.shape[0], 1, device=ray_o.device)

    # Position in model space
    x = torch.addcmul(ray_o, ray_d, t)

    cond = torch.ones_like(t).bool()[:, 0]

    normal = torch.zeros_like(x)
    # This function is in fact differentiable, but we treat it as if it's not, because
    # it evaluates a very long chain of recursive neural networks (essentially a NN with depth of
    # ~1600 layers or so). This is not sustainable in terms of memory use, so we return the final hit
    # locations, where additional quantities (normal, depth, segmentation) can be determined. The
    # gradients will propagate only to these locations.
    with torch.no_grad():

        # cond = nef.grid.query(x)
        # cond = torch.where(cond == -1, 0.0, 1.0).to(torch.bool)

        # print("torch.sum(cond) before")
        # print(torch.sum(cond))

        d = nef(coords=x, channels="sdf")

        dprev = d.clone()

        # If cond is TRUE, then the corresponding ray has not hit yet.
        # OR, the corresponding ray has exit the clipping plane.
        # cond = torch.ones_like(d).bool()[:,0]

        # If miss is TRUE, then the corresponding ray has missed entirely.
        hit = torch.zeros_like(d).byte()
        # print("valid.shape")
        # print(valid.shape)
        # print("valid.shape")
        # print(valid.shape)
        # exit()

        for i in range(num_steps):
            timer.check("start")
            # 1. Check if ray hits.
            # hit = (torch.abs(d) < self._MIN_DIS)[:,0]
            # 2. Check that the sphere tracing is not oscillating
            # hit = hit | (torch.abs((d + dprev) / 2.0) < self._MIN_DIS * 3)[:,0]

            # 3. Check that the ray has not exit the far clipping plane.
            # cond = (torch.abs(t) < self.clamp[1])[:,0]

            hit = (torch.abs(t) < max_dist)[:, 0]

            # 1. not hit surface
            cond = cond & (torch.abs(d) > min_dist)[:, 0]

            # 2. not oscillating
            cond = cond & (torch.abs((d + dprev) / 2.0)
                           > min_dist * 3)[:, 0]

            # 3. not a hit
            cond = cond & hit

            # cond = cond & ~hit

            # If the sum is 0, that means that all rays have hit, or missed.
            if not cond.any():
                break

            # Advance the x, by updating with a new t
            x = torch.where(
                cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)

            # Store the previous distance
            dprev = torch.where(cond.unsqueeze(1), d, dprev)

            # Update the distance to surface at x
            d[cond] = nef(coords=x[cond], channels="sdf") * step_size

            # Update the distance from origin
            t = torch.where(cond.view(cond.shape[0], 1), t+d, t)
            timer.check("end")

    # print("torch.sum(hit) after")
    # print(torch.sum(hit))

    # AABB cull

    hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)
    
    # hit = torch.ones_like(d).byte()[...,0]

    # The function will return
    #  x: the final model-space coordinate of the render
    #  t: the final distance from origin
    #  d: the final distance value from
    #  miss: a vector containing bools of whether each ray was a hit or miss

    if hit.any():
        grad = finitediff_gradient(x[hit], nef.get_forward_function("sdf"))
        _normal = F.normalize(grad, p=2, dim=-1, eps=1e-5)
        normal[hit] = _normal

    return {"x": x, "depth": t, "hit": hit, "normal": normal, "cond": cond, "within_dist": d < min_dist}


def save_mesh(name, verts, faces=None):
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(name)


def load_and_normalize_mesh(mesh_name):
    import trimesh

    mesh = trimesh.load(mesh_name)
    V = torch.tensor(mesh.vertices)

    V_max, _ = torch.max(V, dim=0)
    V_min, _ = torch.min(V, dim=0)
    V_center = (V_max + V_min) / 2.
    V = V - V_center

    # Find the max distance to origin
    max_dist = torch.sqrt(torch.max(torch.sum(V**2, dim=-1)))
    V_scale = 1. / max_dist
    V *= V_scale

    shiften_mesh = trimesh.Trimesh(vertices=V.cpu().numpy(), faces=mesh.faces)

    return shiften_mesh


def marching_cubes(pipeline, output_name, gt_mesh_name):

    gt_mesh = load_and_normalize_mesh(gt_mesh_name)

    from tqdm import tqdm
    # initialize a grid of points to query network with
    grid_size = 512
    pts_x = torch.linspace(-1, 1, grid_size)
    pts_y = torch.linspace(-1, 1, grid_size)
    pts_z = torch.linspace(-1, 1, grid_size)

    all_pts = torch.stack(torch.meshgrid(pts_x, pts_y, pts_z), dim=-1)
    all_pts = all_pts.reshape(-1, 3)

    max_batch = 10000

    est_sdfs = []

    for i in tqdm(range(0, all_pts.shape[0], max_batch)):
        with torch.no_grad():

            # query network with batch of points
            batch = all_pts[i: i + max_batch].cuda()
            batch = batch.unsqueeze(0)
            valid = pipeline.nef.grid.query(batch[0])
            invalid_points = batch[0, valid == -1]
            gt_contains = torch.tensor(
                gt_mesh.contains(invalid_points.cpu().numpy())).cuda()
            gt_points = torch.where(gt_contains, -1.0*torch.ones_like(
                gt_contains), torch.ones_like(gt_contains)).float()
            out = pipeline.nef(coords=batch)['sdf']

            # replace entries of out that are -1 with 1
            out[0, valid == -1, 0] = gt_points

            est_sdfs.append(out.cpu())

    est_sdfs = torch.cat(est_sdfs, dim=1)

    est_sdfs = est_sdfs.reshape(grid_size, grid_size, grid_size)

    from skimage import measure

    verts, faces, normals, values = measure.marching_cubes(
        est_sdfs.cpu().numpy(), level=0.0,
    )

    verts = verts/(grid_size-1)*2-1

    save_mesh(f'output/{output_name}.obj', verts, faces)


if __name__ == "__main__":
    import app_utils
    import torch

    from wisp.trainers import *
    from wisp.config_parser import parse_options, argparse_to_str, get_modules_from_config, \
        get_optimizer_from_config
    from wisp.framework import WispState

    # from wisp.tracers.sdf_tracer import SDFTracer

    # Usual boilerplate
    parser = parse_options(return_parser=True)
    app_utils.add_log_level_flag(parser)
    app_group = parser.add_argument_group('app')
    # Add custom args if needed for app
    args, args_str = argparse_to_str(parser)
    app_utils.default_log_setup(args.log_level)
    pipeline, train_dataset, device = get_modules_from_config(args)
    optim_cls, optim_params = get_optimizer_from_config(args)
    trainer = globals()[args.trainer_type](pipeline, train_dataset, args.epochs, args.batch_size,
                                           optim_cls, args.lr, args.weight_decay,
                                           args.grid_lr_weight, optim_params, args.log_dir, device,
                                           exp_name=args.exp_name, info=args_str, extra_args=vars(
                                               args),
                                           render_every=args.render_every, save_every=args.save_every)
    

    mesh_name = args.dataset_path.split('/')[-3]
    
    
    save_folder = f"/ubc/cs/research/kmyi/dhf/output_multiview/{mesh_name}/part0/num_lods_{args.num_lods}"
    
    # make the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # the output folder should be /ubc/cs/research/kmyi/dhf/output_multiview/{mesh_name}/part0/num_lods_{num_lods}
    
    if args.valid_only:
        trainer.validate()
    else:
        trainer.train()

    x_valid_hit = trace_num_points(pipeline, num_points=10_000_000)
    
    surface_points = trimesh.Trimesh(vertices=x_valid_hit.cpu().numpy(), faces=None)
    surface_points.export(f"{save_folder}/surface_points.obj")

    mesh = load_and_normalize_mesh(args.dataset_path)
    mesh.export(f'{save_folder}/gt_mesh.obj')

    sampled_gt_points = mesh.sample(10_000_000)
    sampled_gt_points = trimesh.Trimesh(vertices=sampled_gt_points, faces=None)
    sampled_gt_points.export(f'{save_folder}/gt_points.obj')