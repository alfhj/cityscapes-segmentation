
import wandb
import torchvision.transforms.functional as transformsF

def upload_image(img, output, label):
    img = img[0].cpu().permute(1, 2, 0)
    output = output[0].cpu().clamp(0, 1)
    label = label[0].cpu()
    # norm: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
    # denorm = img = img * std + mean
    label = transformsF.to_pil_image(label)
    output = transformsF.to_pil_image(output)
    #denorm = lambda x: x * torch.Tensor(stds) + torch.Tensor(means)
    denorm = lambda x: x
    img_denorm = denorm(img).permute(2, 0, 1).clamp(0, 1)
    img_denorm = transformsF.to_pil_image(img_denorm)
    return [wandb.Image(img, caption="1: Input, 2: Output, 3: Label") for img in [img_denorm, output, label]]