import torch
from LongCLIP.model import longclip as clip
from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# #model, preprocess = clip.load("/home/t2f/ResShift_text/ResShift-master/weights/models--BeichenZhang--LongCLIP-B/longclip-B.pt", device=device)
# model, preprocess = clip.load("/home/t2f/Long-CLIP-main/converted_longclip_ft_35.pt", device=device)

MAX_LENGTH = 248

def calculate_clip_loss(model,preprocess, image_list, text_list,device='cuda'):
    assert len(image_list) == len(text_list), "length not same!"

    def tokenize_with_truncation(text, max_length):
        return clip.tokenize(text, context_length=max_length, truncate=True)

    text_inputs = torch.cat([tokenize_with_truncation(text, MAX_LENGTH) for text in text_list], dim=0).to(device)

    image_tensor_list = [preprocess(image).unsqueeze(0) for image in image_list]
    image_inputs = torch.cat(image_tensor_list, dim=0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)

    similarities = (image_features @ text_features.t()).diagonal()
    clip_loss = (1. - similarities / 100)

    return clip_loss
