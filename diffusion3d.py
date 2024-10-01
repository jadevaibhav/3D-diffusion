import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
import math
import os
import deepdish as dd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
from diffusers import DDIMScheduler,DDPMScheduler
from diffusers.training_utils import EMAModel

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Interpolate(nn.Module):
    def __init__(self,  scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self. scale_factor, mode=self.mode)
        return x

def Upsample(dim):
        
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0,mode='nearest'),
        #Interpolate(scale_factor=2.0,mode="nearest"),
        nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1)
                          )

def Downsample(dim):
    return nn.Conv3d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) d x y -> b h c (d x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (d x y) d1 -> b (h d1) d x y", x=h, y=w, d=d)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) d x y -> b h c (d x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (d x y) -> b (h c) d x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        with_time_emb=True,
        resnet_block_groups=8,
        device = device
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv3d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv3d(dim, out_dim, 1)
        )

        if device:
            self.device = device
            self.to(device)

    def forward(self, x, time):
        # Returns the noise prediction from the noisy volume
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        noise_pred = self.final_conv(x)
        return noise_pred
    
class VoxelDataset(Dataset):
  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.voxel_files = [f for f in os.listdir(root_dir) if (f.endswith('.dd') and f.split('_')[4] in ['02691156','03001627'])]

  def __len__(self):
    return len(self.voxel_files)

  def __getitem__(self, idx):
    voxel_path = os.path.join(self.root_dir, self.voxel_files[idx])
    voxel_grid = dd.io.load(voxel_path)['data']
    voxel_tensor = torch.tensor(voxel_grid, dtype=torch.float32)
    
    return voxel_tensor


from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 80
    eval_batch_size = 32  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    lr_warmup_steps = 800
    save_image_epochs = 5
    save_model_epochs = 2
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    input_dir = "3d-ddpm-vanilla"
    output_dir = "3d-ddpm-vanilla"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    # Whether to use Exponential Moving Average for the final model weights.
    use_ema = False #@param {type:"boolean"}

    # The inverse gamma value for the EMA decay.
    ema_inv_gamma = 1.0 #@param {type:"number"}

    # The power value for the EMA decay.
    ema_power = 3 / 4 #@param {type:"raw"}

    # The maximum decay magnitude for EMA.
    ema_max_decay = 0.9999 #@param {type:"number"}

    scheduler_config = {
   
    "beta_schedule": "squaredcos_cap_v2",
    "clip_sample": True,
    "clip_sample_range": 1.0,
    "num_train_timesteps": 1000,
    #"num_inference_steps":1000,
    "prediction_type": "epsilon",
    #"sample_max_value": 1.0,
    #"timestep_spacing":"trailing",
    "rescale_betas_zero_snr":False,
    }
"""{
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": True,
    "clip_sample_range": 1.0,
    "dynamic_thresholding_ratio": 0.995,
    "num_train_timesteps": 1000,
    "prediction_type": "v_prediction",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "thresholding": False,
    "rescale_betas_zero_snr": True,
    }"""

def scheduler_to_device(scheduler,device):
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    if hasattr(scheduler, 'final_alpha_cumprod'):
        scheduler.final_alpha_cumprod = scheduler.final_alpha_cumprod.to(device)

#from diffusers import EMA
def eval(model,config,epoch):
    if not config.use_ema:
        model.eval()
    else:
        model = model.averaged_model
    from diffusers.utils.torch_utils import randn_tensor

    generator = torch.Generator(device=device).manual_seed(config.seed)
    scheduler = DDIMScheduler.from_config(config.scheduler_config,
                                rescale_betas_zero_snr=True, timestep_spacing="trailing",
                                prediction_type = "epsilon")
    #DDPMScheduler.from_config(config.scheduler_config)
    scheduler.set_timesteps(num_inference_steps=50,device=device)
    scheduler_to_device(scheduler,device)

    image = randn_tensor((1,1,config.image_size,config.image_size,config.image_size),device=device, generator=generator)
    #torch.randn((1,1,config.image_size,config.image_size,config.image_size),device=device)
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps[:,None]):
            # 1. predict noise model_output
            model_output = model(image, t)
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = scheduler.step(
                model_output, t, image, eta=0, use_clipped_model_output=True,  generator=generator
            ).prev_sample
    # 
    torch.save(image.detach().cpu().numpy(),os.path.join(config.output_dir, 'gen-32-{ep}ep.pt'.format(ep=epoch)))

### calculate the original sample from velocity pred by model and input noisy image, for v_prd mode
### (see section 2.4 of [Imagen Video](https://imagen.research.google/video/paper.pdf) paper).

def get_orig_from_velocity(model_output,timestep,sample,scheduler):
    alpha_prod_t = scheduler.alphas_cumprod[timestep].view(-1,1,1,1,1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    if scheduler.config.thresholding:
            pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    return pred_original_sample


import wandb
import pathlib
if __name__ == '__main__':
    
    config = TrainingConfig()
    run = wandb.init(project="3d-gen",
                     name=f"experiment_no_ema_{config.output_dir}",config=config.scheduler_config)
    
    ### create output dir
    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True) 
    
    ### dataloader
    dataset = VoxelDataset(root_dir='shape_net_voxel_data_v1')
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, 
                            shuffle=True,num_workers=5,persistent_workers=True,
                            pin_memory=False)
    
    ### loading model
    model = Unet3D(init_dim=32,dim=32,dim_mults=[1,2,4,8])
    if os.path.isfile(os.path.join(config.input_dir, 'model')):
       model.load_state_dict(torch.load(os.path.join(config.input_dir, 'model')))
    #model.to(device)
    ### next experiment use ema model
    if config.use_ema:
        ema_model = EMAModel(
                model.parameters(),
                inv_gamma=config.ema_inv_gamma,
                power=config.ema_power,
                max_value=config.ema_max_decay
            )

    from diffusers.optimization import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )
    
    noise_scheduler = DDPMScheduler.from_config(config.scheduler_config)
    scheduler_to_device(noise_scheduler,device)
    for epoch in range(config.num_epochs):
        model.train()
        with tqdm(dataloader, unit="batch", leave=False) as tepoch:
            for  batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")
                clean_images = batch.to(device).unsqueeze(1)
                # Sample noise that we'll add to the images
                noise = torch.randn(clean_images.shape,device=device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                ### trying to learn rverse process from 700th timestep
                #clean_images = noise_scheduler.add_noise(clean_images, noise, torch.tensor([700],device=device))
                #noise = noisy_images - clean_images
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                ### weighting term of SNR+1
                # weights = 1 + (noise_scheduler.alphas[timesteps])**2/(1 - (noise_scheduler.alphas[timesteps])**2)
                # weights = torch.sqrt(weights).view(-1,1,1,1,1)
                
                with torch.set_grad_enabled(True):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps)

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        #alpha_t = noise_scheduler.alphas[timesteps].view(-1,1,1,1,1)
                        #v_pred = alpha_t*noise - torch.sqrt(1-alpha_t**2)*clean_images
                        v_pred = noise_scheduler.get_velocity(clean_images,noise,timesteps)
                        v_recon = get_orig_from_velocity(noise_pred,timesteps,noisy_images,noise_scheduler)
                        # loss = F.huber_loss(weights*noise_pred, weights*v_pred)
                        loss = F.huber_loss(noise_pred, v_pred)

                    elif noise_scheduler.config.prediction_type == "epsilon":
                        # loss = F.huber_loss(weights*noise_pred, weights*noise)
                        loss = F.huber_loss(noise_pred, noise)
                        v_recon = noisy_images - noise_pred

                    elif noise_scheduler.config.prediction_type == "sample":
                        # loss = F.huber_loss(weights*noise_pred, weights*clean_images)
                        loss = F.huber_loss(noise_pred, clean_images)
                        v_recon = noise_pred

                    ### axis of symmetry loss, where upper and lower halves should be mirror images of each other
                    # loss += F.huber_loss(v_recon[...,:config.image_size//2],
                    #                      torch.flip(v_recon[...,config.image_size//2:],dims=[-1]))

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    if config.use_ema:
                        ema_model.step(model)
                    optimizer.zero_grad()
                    
                tepoch.set_postfix(loss=loss.item())
                wandb.log({"train_mse": loss.item(),
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                           #"ema_decay":ema_model.decay
                            })

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                eval(model,config,epoch)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                torch.save(model.state_dict(), os.path.join(config.output_dir, 'model'))
                #torch.save(ema_model.state_dict(), os.path.join(config.output_dir, 'ema-model'))