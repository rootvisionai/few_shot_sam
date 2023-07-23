# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:00:24 2021

@author: tekin.evrim.ozmermer
"""
import torch
import numpy as np
import torchvision.utils
import tqdm
from server_utils import flatten_feature_map, l2_norm


def binarize_labels(labels):
    num_labels = labels.shape[0]
    unique_labels = np.unique(labels)
    num_unique_labels = len(unique_labels)

    binarized_labels = np.zeros((num_labels, num_unique_labels), dtype=int)

    for i, label in enumerate(labels):
        label_index = np.where(unique_labels == label)[0]
        binarized_labels[i, label_index] = 1

    return torch.from_numpy(binarized_labels).float()


class ExactSolution(torch.nn.Module):
    def __init__(self,
                 embedding_collection,
                 labels_int,
                 threshold,
                 device="cuda"):

        super(ExactSolution, self).__init__()

        self.device = device
        # self.loss_func = torch.nn.CrossEntropyLoss()

        self.embedding_collection = l2_norm(embedding_collection).to(self.device).float()
        self.threshold = threshold
        self.num_classes = len(np.unique(labels_int))
        # self.labels_int = torch.from_numpy(labels_int).to(self.device).long()
        self.labels_bin = binarize_labels(labels_int)
        self.linear = torch.nn.Linear(in_features=self.embedding_collection.shape[1],
                                      out_features=self.labels_bin.shape[1],
                                      bias=False)
        self.solve_exact()
        # self.opt = torch.optim.SGD(momentum=0.9, lr=0.1, params=self.linear.parameters())
        # self.train()
        # self.train_linear()
        self.eval()

    def solve_exact(self):
        collection_inverse = torch.pinverse(self.embedding_collection)
        self.W = torch.matmul(collection_inverse.to(self.device),
                              self.labels_bin.to(self.device))
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(self.W.T)

    def infer(self, query_features):

        with torch.no_grad():

            b, n, h, w = query_features.shape
            query_features = flatten_feature_map(query_features)
            query_features = l2_norm(query_features)[0]
            out = self.forward(query_features.float())

            torchvision.utils.save_image(out[:, 1].reshape(b, h, w).squeeze(0).cpu().float(), "./intermediate_preds.png")

            # apply adaptive threshold
            self.threshold = self.threshold if self.threshold <= out[:, 1].max() else out[:, 1].max()
            print(f"ADAPTIVE THRESHOLD: {self.threshold}")
            out = torch.where(out >= self.threshold, 1, 0)

            # get indexes of maximums
            predictions = out.argmax(dim=-1)

            predictions = predictions.reshape(b, h, w).squeeze(0)
            torchvision.utils.save_image(predictions.cpu().float(), "./intermediate_mask.png")

        return predictions

    def train_linear(self):
        pbar = tqdm.tqdm(range(0, 200))
        for epoch in pbar:
            out = self.linear(self.embedding_collection)
            loss = self.loss_func(out, self.labels_int)
            pbar.set_description(f"EPOCH: {epoch} | LOSS: {loss.item()}")

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def forward(self, embedding):
        out = self.linear(embedding)
        out = torch.where(out > 1, 2-out, out)
        out = torch.nn.functional.softmax(out, dim=-1)
        return out
