#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def compute_jaccard_distance(features=None, cam_labels=None, epoch=None, args=None, print_flag=True):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(cam_labels, torch.Tensor):
        cam_labels = cam_labels.cpu().numpy()

    k1, k2 = args.k1, args.k2
    ckrnns, k1_intra, k1_inter = args.ckrnns, args.k1_intra, args.k1_inter
    clqe, k2_intra, k2_inter = args.clqe, args.k2_intra, args.k2_inter

    if ckrnns and clqe:
        mode = f"EPOCH[{epoch}] [CAJaccard (CKRNNS + CLQE)]"
    elif ckrnns and not clqe:
        mode = f"EPOCH[{epoch}] [CAJaccard (CKRNNS + LQE)]"
    elif not ckrnns and clqe:
        mode = f"EPOCH[{epoch}] [CAJaccard (KRNNS + CLQE)]"
    else:
        mode = f"EPOCH[{epoch}] [Jaccard (KRNNS + LQE)]"
    print(mode)

    N = features.shape[0]
    mat_type =np.float32

    # cosine -> eculidean
    original_dist = 2 - 2 * np.matmul(features, features.T)

    cam_mask = (cam_labels.reshape(-1, 1) == cam_labels.reshape(1, -1))
    cam_diff = original_dist[np.triu(~cam_mask, k=1)].mean() - original_dist[np.triu(cam_mask, k=1)].mean()
    print('Camera difference: {:.2f}'.format(cam_diff))



    if ckrnns or clqe:
        inter_rank = np.argpartition(original_dist + 999.0 * cam_mask, range(k1_inter + 2))
        intra_rank = np.argpartition(original_dist + 999.0 * (~cam_mask), range(k1_intra + 2))
    global_rank = np.argpartition(original_dist, range(k1 + 2))
    #           KRNNs/CKRNNs          #
    if ckrnns:
        print(f"EPOCH[{epoch}] [CKRNNs] PARAMS: k1_intra: {k1_intra}, k1_inter: {k1_inter}")
    else:
        print(f"EPOCH[{epoch}] [KRNNs] PARAMS: k1: {k1}")

    if ckrnns:
        nn_inter = [k_reciprocal_neigh(inter_rank, i, k1_inter) for i in range(N)]
        nn_intra = [k_reciprocal_neigh(intra_rank, i, k1_intra) for i in range(N)]
        nn_k1 = [np.union1d(nn_intra[i], nn_inter[i]) for i in range(N)]
    else:
        nn_k1 = [k_reciprocal_neigh(global_rank, i, k1) for i in range(N)]
        nn_k1_half = [k_reciprocal_neigh(global_rank, i, int(np.around(k1 / 2))) for i in range(N)]

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index

        # Jaccard recall
        if not ckrnns:
            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = nn_k1_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)


        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = torch.from_numpy(original_dist[i][k_reciprocal_expansion_index]).unsqueeze(0)
        V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()
    #            LQE/CLQE          #
    ################################
    # warmup
    if epoch == 0:
        print("Warm-up...")
        k2_intra, k2_inter = 3, 3
    if clqe:
        print(f"EPOCH[{epoch}] [CLQE] PARAMS: k2_intra: {k2_intra}, k2_inter: {k2_inter}")
    else:
        print(f"EPOCH[{epoch}] [LQE] PARAMS: k2: {k2}")


    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            if clqe:
                k2nn = np.append(intra_rank[i, :k2_intra], inter_rank[i, :k2_inter])
            else:
                k2nn = global_rank[i, :k2]
            V_qe[i, :] = np.mean(V[k2nn, :], axis=0)
        V = V_qe

    jaccard_dist = v2jaccard(V, N, mat_type)

    print("Distance computing time cost: {}".format(time.time() - end))
    return jaccard_dist

def v2jaccard(V, N, mat_type):
    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    return jaccard_dist



@torch.no_grad()
def compute_ranked_list(features, k=20, search_option=0, fp16=False, verbose=True):

    end = time.time()
    if verbose:
        print("Computing ranked list...")

    if search_option < 3:
        torch.cuda.empty_cache()
        features = features.cuda().detach()

    ngpus = faiss.get_num_gpus()

    if search_option == 0:
        # Faiss Search + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, features, features, k+1)
        initial_rank = initial_rank.cpu().numpy()

    elif search_option == 1:
        # Faiss Search + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, features, k+1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()

    elif search_option == 2:
        # PyTorch Search + PyTorch CUDA Tensors
        torch.cuda.empty_cache()
        features = features.cuda().detach()
        dist_m = compute_euclidean_distance(features, cuda=True)
        initial_rank = torch.argsort(dist_m, dim=1)
        initial_rank = initial_rank.cpu().numpy()

    else:
        # Numpy Search (CPU)
        torch.cuda.empty_cache()
        features = features.cuda().detach()
        dist_m = compute_euclidean_distance(features, cuda=False)
        initial_rank = np.argsort(dist_m.cpu().numpy(), axis=1)
        features = features.cpu()

    features = features.cpu()
    if verbose:
        print("Ranked list computing time cost: {}".format(time.time() - end))

    return initial_rank[:, 1:k+1]

@torch.no_grad()
def compute_euclidean_distance(features, others=None, cuda=False):
    if others is None:
        if cuda:
            features = features.cuda()

        n = features.size(0)
        x = features.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        del features

    else:
        if cuda:
            features = features.cuda()
            others = others.cuda()

        m, n = features.size(0), others.size(0)
        dist_m = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n) +\
                 torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist_m.addmm_(features, others.t(), beta=1, alpha=-2)
        del features, others

    return dist_m
