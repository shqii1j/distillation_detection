import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./save/generation/')
    parser.add_argument('--models', type=str, nargs='+', default=['bk_sdm_v2_base', 'bk_sdm_v2_small', 'bk_sdm_v2_tiny', 'amd', 'amd_pixart', 'dmd2', 'sdxl_1step', 'sdxl_2step', 'sdxl', 'sdxl_8step'])
    parser.add_argument('--score', type=str, default='clip', choices=['clip', 'dino', 'lpips', 'mmdfuse', 'acs', 'cka'])
    parser.add_argument('--input_size', type=int, default=50)
    parser.add_argument('--add_name', type=str, default='', help="The default is our setting")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    '''Load scores'''
    scores = [[] for _ in range(len(args.models))]
    for i, name in enumerate(args.models):
        save_path = os.path.join(args.path, name, args.add_name +f'bz{args.input_size}' + '_scores.pt')
        print('Loading:', save_path)
        data = torch.load(save_path, weights_only=False)
        data = data[args.score]
        if args.score in ['clip', 'dino', 'lpips']:
            data = np.array(data).reshape(len(data), args.input_size, -1).mean(axis=1)
        scores[i] = data

    '''Compute accuracy and AUC'''
    scores = np.array(scores)
    teacher_map = {'bk_sdm_v2_base': 0, 'bk_sdm_v2_small': 0, 'bk_sdm_v2_tiny': 0, 'amd': 0, 'amd_pixart': 2, 'dmd2': 1, 'sdxl_1step': 1, 'sdxl_2step': 1, 'sdxl': 1, 'sdxl_8step': 1}
    target_candidate_indices = np.array([teacher_map[name] for name in args.models])
    if args.score in ['lpips']:
        tar_indices = np.argmin(scores, axis=1)
    else:
        tar_indices = np.argmax(scores, axis=1)
    correct = (tar_indices == target_candidate_indices[:, None]).astype(int)
    acc_per_run = correct.mean(axis=0)

    aucs = []
    for i in range(scores.shape[-1]):
        if args.score in ['lpips']:
            scores = 1 / scores[:, :, i]
        else:
            scores = scores[:, :, i]
        scores = F.softmax(torch.tensor(scores), dim=1).numpy()
        labels = target_candidate_indices
        aucs.append(roc_auc_score(labels, scores, multi_class='ovr').item())
    aucs = np.array(aucs)

    return acc_per_run.mean().item(), acc_per_run.std().item() , aucs.mean().item(), aucs.std().item()

if __name__ == "__main__":
    acc_mean, acc_std, auc_mean, auc_std = main()
    print(f"Accuracy: {acc_mean*100:.2f} ± {acc_std*100:.2f}%, AUC: {auc_mean*100:.2f} ± {auc_std*100:.2f}%")