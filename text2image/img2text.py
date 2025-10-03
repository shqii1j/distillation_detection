import torch
import argparse
import os
from transformers import pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a generating caption script.")
    parser.add_argument("--student_model_names", type=str, nargs='+', 
                        default=['bk_sdm_v2_base', 'bk_sdm_v2_small', 'bk_sdm_v2_tiny', 
                                 'amd', 'dmd2_1step', 'dmd2', 'sdxl_1step', 'sdxl_2step', 
                                 'sdxl', 'sdxl_8step'],
                        help="List of student model names for generating data.") 
    
    parser.add_argument("--save_path", type=str, default='./save/generation', help="Path to save generated captions.")
    parser.add_argument("--input_data", type=str, default='empty_str', help="Type of input images")
    parser.add_argument("--captioning_model", type=str, default='blip_base', help="Captioning model to use")

    return parser.parse_args()

def get_captioning_model(model_name):
    if model_name == 'blip_base':
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    elif model_name == 'gpt2':
        pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    else:
        raise ValueError(f"Unknown captioning model: {model_name}")
    return pipe

def main():
    args = parse_args()
    for student_name in args.student_model_names:
        cap_model = get_captioning_model(args.captioning_model)
        if args.captioning_model == 'blip_base':
            cap_add = 'blip_'
        elif args.captioning_model == 'gpt2':
            cap_add = 'gpt2_'
        
        if args.input_data == 'empty_str':
            add_name = 'empty_'
        

        captions = {"image_path": [], "caption": []}
        
        img_dir = os.path.join(args.save_path, student_name, args.input_data)
        image_paths = [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith('.png')
        ]

        decoded = cap_model(image_paths, batch_size=8)  # Adjust batch_size if needed
        captions['image_path'] = image_paths
        captions['caption'] = [i[0]['generated_text'] for i in decoded]
        torch.save(captions, os.path.join(args.save_path, student_name, add_name + f'{add_name}{cap_add}captions.pt'))
        print(f"Captions saved for {student_name} in {args.save_path} as {add_name}{cap_add}cap.pt'")

if __name__ == "__main__":
    main()