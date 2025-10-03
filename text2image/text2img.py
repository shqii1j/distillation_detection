import torch
from torchvision import transforms
import argparse
import os
from PIL import Image
from torch.utils.data import DataLoader

from _utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--student_model_names", type=str, nargs='+', default=['bk_sdm_v2_base', 'bk_sdm_v2_small', 'bk_sdm_v2_tiny', \
                                                                               'amd', 'dmd2', 'sdxl_1step', \
                                                                               'sdxl_2step', 'sdxl', 'sdxl_8step', 'amd_pixart']) 
    parser.add_argument("--teacher_model_names", type=str, nargs='*', default=['tch_sd', 'tch_sdxl', 'tch_pixart'])
    parser.add_argument("--captioning_model", type=str, default='blip_base')
    parser.add_argument("--save_path", type=str, default='./save/generation')
    parser.add_argument("--input_size", type=int, default=10)
    parser.add_argument("--metrics", type=str, nargs='+', default=['clip', 'dino', 'lpips', 'cka', 'acs', 'mmdfuse'])
    parser.add_argument("--input_data", type=str, default='empty_str')
    parser.add_argument("--save_images", action='store_true')
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--dist_plot', action='store_true')
    parser.add_argument('--acc_table', action='store_true')
    parser.add_argument('--auc', action='store_true')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    cap_model = get_captioning_model(args.captioning_model)
    if args.captioning_model == 'blip_base':
        cap_add = 'blip_'
    elif args.captioning_model == 'gpt2':
        cap_add = 'gpt2_'

    dist_func_l = []
    metric_kwargs_l = []
    is_set_score = []
    for metric in args.metrics:
        dist_func, metric_kwargs = get_compute_distance_func(metric)
        dist_func_l.append(dist_func)
        metric_kwargs_l.append(metric_kwargs)
        if metric in ['cka', 'acs', 'mmdfuse']:
            is_set_score.append(True)
        else:
            is_set_score.append(False)

    if args.input_data == 'empty_str':
        add_name = ''
    elif args.input_data == 'empty_blip_cap':
        add_name = 'empty_blip_'
    elif args.input_data == 'empty_gpt2_cap':
        add_name = 'empty_gpt2_'
    else:
        AssertionError(f"Unknown input data type: {args.input_data}")
    
    add_name = args.input_data + '_bz' + str(args.input_size) + '_'
    
    scores_l = []

    for i, model_name in enumerate(args.student_model_names):
        setup_seeds(42)
        save_path = os.path.join(args.save_path, model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_args = get_model_args(model_name)
        model = load_model(model_args)

        scores = {metric: [[] for _ in range(len(args.teacher_model_names))] for metric in args.metrics}
        
        if args.save_images:
            os.makedirs(os.path.join(save_path, args.input_data), exist_ok=True)
        
        if args.input_data in ['empty_blip_cap', 'empty_gpt2_cap']:
            data = torch.load(os.path.join(save_path, args.input_data + '.pt'))
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5]),
                                            ])
            dataset = ImageTextDataset(data, transform=transform)
            print('The first 10 captions are:', dataset.captions[:11])
            dataloader = DataLoader(dataset, batch_size=args.input_size, shuffle=False, num_workers=4)
            iter_data = iter(dataloader)
        
        for i_run in range(10):
            if args.input_data == 'empty_str':
                prompt = [''] * args.input_size
                prompt_embeds = None
            elif args.input_data in ['empty_gpt2_cap', 'empty_blip_cap']:
                batch = next(iter_data)
                prompt = list(batch[1])
                prompt_embeds = None
            latents = torch.randn(args.input_size, *model_args['image_shape'], dtype=torch.float16).to('cuda') 
            print(f"{model_name}: {i_run} / 10")


            # TODO: load images if already exist
            if os.path.exists(os.path.join(save_path, args.input_data, f'{(i_run + 1) * args.input_size - 1}.png')):
                images1 = []
                for idx in range(args.input_size):
                    image_path = os.path.join(save_path, args.input_data, f'{i_run * args.input_size + idx}.png')
                    images1.append(Image.open(image_path).convert("RGB"))
            else:           
                if 'timesteps' in model_args:
                    timesteps = model_args['timesteps']
                    num_inference_steps = len(timesteps)
                    images1= model(prompt=prompt, num_inference_steps=num_inference_steps, timesteps=timesteps, guidance_scale=0, latents=latents)
                else:
                    images1= model(prompt=prompt, guidance_scale=0, latents=latents)
                images1 = images1[0]
                
                # Save images
                for idx in range(len(images1)):
                    image_path = os.path.join(save_path, args.input_data, f'{i_run * args.input_size + idx}.png')
                    images1[idx].save(image_path)
            
            # Calculate scores
            for j, teacher_model_name in enumerate(args.teacher_model_names):
                # TODO: load images if already exist
                if os.path.exists(os.path.join(save_path, args.input_data, 'T_' + teacher_model_name, f'{(i_run + 1) * args.input_size - 1}.png')):
                    images2 = []
                    for idx in range(args.input_size):
                        image_path = os.path.join(save_path, args.input_data, 'T_' + teacher_model_name, f'{i_run * args.input_size + idx}.png')
                        images2.append(Image.open(image_path).convert("RGB"))
                else:
                    tch_args = get_model_args(teacher_model_name)
                    tch_model = load_model(tch_args)
                    images2 = tch_model(prompt=prompt, guidance_scale=2, latents=latents)
                    images2 = images2[0]
                
                    for idx, (img1, img2) in enumerate(zip(images1, images2)):
                        image_path = os.path.join(save_path, args.input_data, 'T_' + teacher_model_name, f'{i_run * args.input_size + idx}.png')
                        if not os.path.exists(os.path.join(save_path, args.input_data, 'T_' + teacher_model_name)):
                            os.makedirs(os.path.join(save_path, args.input_data, 'T_' + teacher_model_name))
                        img2.save(image_path)
                    continue
                
                for k, (dist_func, metric_kwargs) in enumerate(zip(dist_func_l, metric_kwargs_l)):
                    if is_set_score[k]:
                        if args.input_size > 1:
                            dist = dist_func(images1, images2, **metric_kwargs)
                            scores[args.metrics[k]][j].append(dist)
                    else:
                        for img1, img2 in zip(images1, images2):
                            img1 = transforms.ToTensor()(img1).to('cuda')
                            if img1.shape[1] != 512:
                                img1 = transforms.Resize((512, 512))(img1)
                            img2 = transforms.ToTensor()(img2).to('cuda')
                            if img2.shape[1] != 512:
                                img2 = transforms.Resize((512, 512))(img2)
                            dist = dist_func(img1, img2, **metric_kwargs)
                            scores[args.metrics[k]][j].append(dist)
            print("\n")
        
        # Save scores
        torch.save(scores, os.path.join(save_path, add_name + 'scores.pt'))
        scores_l.append(scores)
        print(f"Scores saved for {model_name} in {save_path} as {add_name}scores.pt")