from diffusion3d import *
from diffusers.utils.torch_utils import randn_tensor

def scheduler_to_device(scheduler,device):
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.final_alpha_cumprod = scheduler.final_alpha_cumprod.to(device)



if __name__ == '__main__':

    config = TrainingConfig()
    model = Unet3D(init_dim=32,dim=32,dim_mults=[1,2,4,8])
    model.load_state_dict(torch.load(os.path.join(config.output_dir, 'model')))
    model.eval()

    bs = 16
    generator = torch.Generator(device=device)
    scheduler = DDIMScheduler.from_config(config.scheduler_config,rescale_betas_zero_snr=True,
                                            timestep_spacing="trailing",prediction_type = "v_prediction")
    scheduler.set_timesteps(num_inference_steps=50,device=device)
    scheduler_to_device(scheduler,device)

    with torch.no_grad():
        image = randn_tensor((bs,1,config.image_size,config.image_size,config.image_size),device=device, generator=generator)
        #torch.randn((1,1,config.image_size,config.image_size,config.image_size),device=device)
        for t in tqdm(scheduler.timesteps[:,None]):
            # 1. predict noise model_output
            model_output = model(image, t)
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = scheduler.step(
                model_output, t, image, eta=0, use_clipped_model_output=True, generator=generator
            ).prev_sample

    torch.save(image.detach().cpu().numpy(),os.path.join(config.output_dir, 'gen-32-ddim-final.pt'))
