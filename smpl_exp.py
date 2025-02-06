from typing import List, Optional, Union
import torch
import cv2
from diffusers.utils import export_to_video

# Util function for loading meshes
from functools import cache

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)
from SMPL.SMPL_master.smpl_torch import SMPLModel
import einops
import types

# add path for demo utils functions
import os

import torch
from diffusers import (
    AutoencoderKLLTXVideo,
    LTXPipeline,
)
import PIL.Image
from diffusers.utils import export_to_video
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
device = torch.device("cpu")

pipe: LTXPipeline = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16,
    cache_dir="/home/dcor/giladd/arm_exps/cache",
)
pipe.to(device)

pipe._callback_tensor_inputs = [
    "latents",
    "prompt_embeds",
    "negative_prompt_embeds",
    "noise_pred",
]


# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
R, T = look_at_view_transform(2.7, 0, 180)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
# the difference between naive and coarse-to-fine rasterization.
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])


renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
)

import PIL.Image
import numpy as np

target_image = PIL.Image.open("/home/dcor/giladd/arm_exps/SMPL/smpl_torch.png").resize(
    (512, 512)
)
target_image = torch.tensor(np.array(target_image) / 255.0).to(device).float()

iterations = 1
number_of_frames = 10
pose_size = 72
beta_size = 10
trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(device)
betas = (
    torch.from_numpy((np.random.rand(beta_size) - 0.5) * 0.06)
    .type(torch.float64)
    .to(device)
)
np.random.seed(9607)
prompt = "A 3D render of a human body on a blank background. The human raises its hand upwards and then lowers them back down"
autoencoder: AutoencoderKLLTXVideo = pipe.vae


pipe.encode_prompt = types.MethodType(cache(pipe.encode_prompt), pipe)
scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler


def set_timesteps_explicit(
    self: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: int = None,
    device: Union[str, torch.device] = None,
    sigmas: Optional[List[float]] = None,
    mu: Optional[float] = None,
    timesteps: Optional[List[torch.Tensor]] = None,
):
    if timesteps is not None:
        return timesteps

    return self.set_timesteps_prev(
        num_inference_steps=num_inference_steps,
        sigmas=sigmas,
        mu=mu,
        device=device,
    )


scheduler.set_timesteps_prev = types.MethodType(scheduler.set_timesteps, scheduler)
scheduler.set_timesteps = types.MethodType(set_timesteps_explicit, scheduler)

with torch.enable_grad():

    pose_0 = (
        torch.from_numpy((np.random.rand(pose_size) - 0.5) * 0.4)
        .type(torch.float64)
        .to(device)
    )

    poses = [torch.clone(pose_0) for _ in range(number_of_frames)]
    for pose in poses:
        pose.requires_grad = True

    smpl_model = SMPLModel(device=device)
    faces = torch.tensor(smpl_model.faces.astype(np.int32)).to(device)

    optimizer = torch.optim.Adam(poses, lr=0.1)
    for i in range(iterations):
        frames_i = []
        for t in range(number_of_frames):
            pose = poses[t]
            verts = smpl_model.forward(betas, pose, torch.zeros(3).to(device))
            mesh = Meshes(verts=[verts.to(torch.float32)], faces=[faces])
            dummy_texture = TexturesUV(
                maps=torch.ones((512, 512, 3), device=device)[None],
                faces_uvs=torch.ones_like(mesh.faces_packed(), device=device)[None],
                verts_uvs=torch.ones_like(mesh.verts_packed(), device=device)[None][
                    :, :, :2
                ],
            )
            mesh.textures = dummy_texture
            frame_t = renderer.forward(mesh)
            frames_i.append(frame_t)

        video_i = torch.stack(frames_i).to(torch.bfloat16)[
            :, :, :, :, :3
        ]  # RGBA -> RGB

        video_i = einops.rearrange(video_i, "T B H W C -> B C T H W")

        latent_video_i = autoencoder.encode(video_i).latent_dist.mode()
        latent_video_i = pipe._normalize_latents(latent_video_i)

        with torch.no_grad():
            epsilon = torch.randn_like(latent_video_i)
            timestep = (
                torch.randint(
                    low=50,
                    high=1000 - 1,  # Avoid the highest timestep.
                    size=(1,),
                    device=latent_video_i.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .to(torch.int32)
            )

            noised_latent_video_i = scheduler.scale_noise(
                latent_video_i, timestep, epsilon
            )

            def callback_on_step_end(self, i, t, callback_kwargs):
                global noise_pred_cache
                noise_pred_cache = callback_kwargs["noise_pred"]

            pipe.__call__(
                latents=noised_latent_video_i,
                guidance_scale=2,
                timesteps=[timestep],
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["noise_pred"],
                prompt=prompt,
            )

            noise_pred = noise_pred_cache

        # poses_grad = torch.autograd.grad(noised_latent_video_i, torch.stack(poses))
        # poses_grad = torch.matmul((noise_pred - epsilon), poses_grad)

        poses_sds_loss: torch.Tensor = (noise_pred - epsilon) * latent_video_i
        poses_sds_loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

        video_name = f"results/res_{i}.avi"

        video_np = [
            (image.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
            for image in frames_i
        ]
        os.makedirs("results", exist_ok=True)
        export_to_video(video_np, f"results/res_{i}.mp4", fps=25)
