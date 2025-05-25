import time
import numpy as np
import random
import torch
import pickle

from args import get_args
from graph import graph_reader, graph_sampler
from train import scratch, pretrain


seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    args = get_args()
    
    print(f"=============\nArgs: batch_size={args.batch_size}, dropout={args.dropout}, lr={args.lr}")

    st = time.time()
    feats, labels, dists, edges, types, relations, feat_dims, label_dim = graph_reader(args)
#     print(f"main.py: {types}")
    trial = 1
    et = time.time()
    print('Data is read: %.1fs' % (et - st))
    
    st = time.time()
    source_data, target_data = graph_sampler(args, feats, labels, edges, label_dim, dists)
    et = time.time()
    print('Data preprocessed: %.1fs' % (et - st))

    summary = np.zeros((trial, 7))
    for t in range(trial):
#         src_loss, src_micro_acc, src_macro_acc, src_match_loss, \
#             trg_loss1, trg_micro_acc1, trg_macro_acc1 = pretrain(args, types, relations, feat_dims, label_dim, source_data, target_data)
        src_loss, src_micro_acc, src_macro_acc, src_match_loss = pretrain(args, types, relations, feat_dims, label_dim, source_data, target_data)
#         summary[t] = np.array([src_loss, src_micro_acc, src_macro_acc, src_match_loss, trg_loss1, trg_micro_acc1, trg_macro_acc1])

#     avg = np.average(summary, axis=0)
#     std = np.std(summary, axis=0)

#     print('Average: ', avg)
#     print('Standard Deviation: ', std)


if __name__ == "__main__":
    main()
