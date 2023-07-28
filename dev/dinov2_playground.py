import torch

device = "cuda"

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
dinov2_vits14.eval()
dinov2_vits14.to(device)

from PIL import Image
import torchvision

image = Image.open("../query_images/bmw_i20_s.jpg")
image = torchvision.transforms.functional.resize(image, (224, 224))
image = torchvision.transforms.functional.to_tensor(image)

# attentions = dinov2_vits14.get_last_selfattention(image.unsqueeze(0).to(device)).cpu()
output = dinov2_vits14(image.unsqueeze(0).to(device))
print(output.shape)