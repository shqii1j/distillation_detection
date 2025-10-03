import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoModel, AutoImageProcessor
from transformers import pipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, PixArtSigmaPipeline
from diffusers import EulerDiscreteScheduler, LCMScheduler, DDPMScheduler
from diffusers import UNet2DConditionModel
from diffusers.utils import load_file, hf_hub_download
from PIL import Image
import lpips
import yaml
import math
from scipy.linalg import orthogonal_procrustes
from mmdfuse import mmdfuse
from jax import random as jr



def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMetric:
    def __init__(self):
        self.reset()

    def update(self, value, n=1):
        self.total += value * n
        self.count += n
        self.avg = self.total / self.count

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.avg = 0.0


''' Loading dataloader'''
class ImageTextDataset(Dataset):
    def __init__(self, data_dict, transform=None, encoder=None):
        self.image_paths = data_dict['image_path']
        self.captions = data_dict['caption']
        self.encoder = encoder
        if encoder is not None:
            self.text_embeddings = [self.encoder(prompt=caption, device='cuda', num_images_per_prompt=1, do_classifier_free_guidance=True)[0].detach().cpu().squeeze(0) for caption in self.captions]
        self.transform = transform
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and optionally transform image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get text embedding
        if self.encoder is not None:
            text_embedding = self.text_embeddings[idx]
            return image, text_embedding
        else:
            return image, self.captions[idx]

    
''' Loading models'''
def get_pipe(pipeline, model_id, unet=None):
    if pipeline == 'stable_diffusion':
        if unet is None:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16).to("cuda")
    elif pipeline == 'stable_diffusion_xl':
        if unet is None:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16).to("cuda")
    elif pipeline == 'pixart-sigma':
        pipe = PixArtSigmaPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        if unet is not None:
            AssertionError("PixArtSigmaPipeline does not support custom UNet")
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    return pipe

def get_scheduler(pipe, scheduler_name):
    if scheduler_name == "euler_discrete":
        return EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif scheduler_name == "lcms":
       return LCMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "ddpm":
        return DDPMScheduler.from_config(pipe.scheduler.config, subfolder="scheduler")
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def get_model_args(model_name):
    mapping_dict = {'dmd2': 'config/dmd2.yaml',
                    'amd': 'config/amd.yaml',
                    'amd_pixart': 'config/amd_pixart.yaml',
                    'sdxl_1step': 'config/sdxl_1step.yaml',
                    'sdxl_2step': 'config/sdxl_2step.yaml',
                    'sdxl_4step': 'config/sdxl_4step.yaml',
                    'sdxl_8step': 'config/sdxl_8step.yaml',
                    'bk_sdm_v2_base': 'config/bk_sdm_v2_base.yaml',
                    'bk_sdm_v2_small': 'config/bk_sdm_v2_small.yaml',
                    'bk_sdm_v2_tiny': 'config/bk_sdm_v2_tiny.yaml',
                    'tch_sdxl': 'config/tch_sdxl.yaml',
                    'tch_sd': 'config/tch_sd.yaml',
                    'tch_pixart': 'config/tch_pixart.yaml',}
    if model_name not in mapping_dict:
        if os.path.exists('config/' + model_name + '.yaml'):
            with open('config/' + model_name + '.yaml', "r") as f:
                args = yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    else:
        with open(mapping_dict[model_name], "r") as f:
            args = yaml.safe_load(f)
    return args

def get_teacher_id(model_name, teacher_names):
    mapping_dict = {'bk_sdm_v2_base': 'tch_sd',
                    'bk_sdm_v2_small': 'tch_sd',
                    'bk_sdm_v2_tiny': 'tch_sd',
                    'amd': 'tch_sd',
                    'amd_pixart': 'tch_pixart',
                    'sdxl_1step': 'tch_sdxl',
                    'sdxl_2step': 'tch_sdxl',
                    'sdxl_4step': 'tch_sdxl',
                    'sdxl_8step': 'tch_sdxl',
                    'dmd2': 'tch_sdxl',}

    tch_id = dict({v:i for i, v in enumerate(teacher_names)})
    return tch_id[mapping_dict[model_name]]

def load_model(model_args):
    base_model_id = model_args['base_model_id']
    unet = None
    if 'unet' in model_args:
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
        repo_name = model_args['unet']['repo_name']
        ckpt_name = model_args['unet']['ckpt_name']
        if '.safetensors' in ckpt_name:
            unet.load_state_dict(load_file(hf_hub_download(repo_name, ckpt_name), device="cuda"))
        else:
            unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda"))
    
    pipe = get_pipe(model_args['pipeline'], base_model_id, unet)

    if 'transformer' in model_args:
        repo_name = model_args['transformer']['repo_name']
        ckpt_name = model_args['transformer']['ckpt_name']
        if '.safetensors' in ckpt_name:
            pipe.transformer.load_state_dict(load_file(hf_hub_download(repo_name, ckpt_name), device="cuda"))
        else:
            pipe.transformer.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda"))
    
    if 'scheduler' in model_args:
        pipe.scheduler = get_scheduler(pipe, model_args['scheduler'])

    return pipe


''' Calculating statistics / distances'''
def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)

def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX

def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def aligned_cosine_similarity(X, Y):
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    R, _ = orthogonal_procrustes(Yc, Xc)
    Y_aligned = Yc.dot(R)
    dot = np.sum(Xc * Y_aligned, axis=1)
    norm = np.linalg.norm(Xc, axis=1) * np.linalg.norm(Y_aligned, axis=1)
    return (dot / norm).mean()

def load_image(path, transform):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to('cuda')

def compute_clip_similarity(img1, img2, **kwargs):
    clip_model, clip_processor = kwargs['clip_model'], kwargs['clip_processor']
    inputs1 = clip_processor(images=img1, return_tensors='pt').to('cuda')
    inputs2 = clip_processor(images=img2, return_tensors='pt').to('cuda')
    with torch.no_grad():
        feat1 = clip_model.get_image_features(**inputs1)
        feat2 = clip_model.get_image_features(**inputs2)
    feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
    feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
    return F.cosine_similarity(feat1, feat2).item()

def compute_dino_similarity(img1, img2, **kwargs):
    dino_model, dino_processor = kwargs['dino_model'], kwargs['dino_processor']
    inputs1 = dino_processor(images=img1, return_tensors='pt').to('cuda')
    inputs2 = dino_processor(images=img2, return_tensors='pt').to('cuda')
    with torch.no_grad():
        feat1 = dino_model(**inputs1).last_hidden_state.mean(1)
        feat2 = dino_model(**inputs2).last_hidden_state.mean(1)
    feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
    feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
    return F.cosine_similarity(feat1, feat2).item()

def compute_lpips_distance(img1, img2, **kwargs):
    lpips_loss = kwargs['lpips_loss']
    with torch.no_grad():
        dist = lpips_loss(img1, img2)
    return dist.item()

def compute_cka_similarity(imgs1, imgs2, **kwargs):
    model, processor = kwargs['model'], kwargs['processor']
    inputs1 = processor(images=[img1 for img1 in imgs1], return_tensors='pt').to('cuda')
    inputs2 = processor(images=[img2 for img2 in imgs2], return_tensors='pt').to('cuda')
    with torch.no_grad():
        feat1 = model.get_image_features(**inputs1)
        feat2 = model.get_image_features(**inputs2)
    feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
    feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)

    return kernel_CKA(feat1.cpu().numpy(), feat2.cpu().numpy()).item()

def compute_acs_similarity(imgs1, imgs2, **kwargs):
    model, processor = kwargs['model'], kwargs['processor']
    inputs1 = processor(images=[img1 for img1 in imgs1], return_tensors='pt').to('cuda')
    inputs2 = processor(images=[img2 for img2 in imgs2], return_tensors='pt').to('cuda')
    with torch.no_grad():
        feat1 = model.get_image_features(**inputs1)
        feat2 = model.get_image_features(**inputs2)
    feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
    feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
    return aligned_cosine_similarity(feat1.cpu().numpy(), feat2.cpu().numpy()).mean().item()

def compute_mmdfuse(imgs1, imgs2, **kwargs):
    model, processor = kwargs['model'], kwargs['processor']
    inputs1 = processor(images=[img1 for img1 in imgs1], return_tensors='pt').to('cuda')
    inputs2 = processor(images=[img2 for img2 in imgs2], return_tensors='pt').to('cuda')
    with torch.no_grad():
        feat1 = model.get_image_features(**inputs1)
        feat2 = model.get_image_features(**inputs2)
    feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
    feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
    key = jr.PRNGKey(42)
    key, subkey = jr.split(key)
    _, p_value = mmdfuse(feat1.cpu().numpy(), feat2.cpu().numpy(), subkey, return_p_val=True)
    return p_value.item()

def get_compute_distance_func(metric):
    if metric == 'clip':
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        return compute_clip_similarity, {'clip_model': clip_model, 'clip_processor': clip_processor}
    elif metric == 'dino':
        dino_model = AutoModel.from_pretrained("facebook/dino-vitb16").to('cuda')
        dino_processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb16")
        return compute_dino_similarity, {'dino_model': dino_model, 'dino_processor': dino_processor}
    elif metric == 'lpips':
        lpips_loss = lpips.LPIPS(net='alex').to('cuda')
        return compute_lpips_distance, {'lpips_loss': lpips_loss}
    elif metric == 'cka':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        return compute_cka_similarity, {'model': model, 'processor': processor}
    elif metric == 'acs':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        return compute_acs_similarity, {'model': model, 'processor': processor}
    elif metric == 'mmdfuse':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        return compute_mmdfuse, {'model': model, 'processor': processor}
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
def get_captioning_model(model_name):
    if model_name == 'blip_base':
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    elif model_name == 'gpt2':
        pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    else:
        raise ValueError(f"Unknown captioning model: {model_name}")
    return pipe
