# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:00:24 2021

@author: tekin.evrim.ozmermer
"""
import torch
import numpy as np
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
        self.embedding_collection = embedding_collection
        self.threshold = threshold
        self.num_classes = len(np.unique(labels_int))
        self.labels_bin = binarize_labels(labels_int)
        self.linear = torch.nn.Linear(in_features=self.embedding_collection.shape[1],
                                      out_features=self.labels_bin.shape[1],
                                      bias=False)
        self.solve_exact()
        self.eval()

    def solve_exact(self):
        collection_inverse = torch.pinverse(l2_norm(self.embedding_collection)).float()
        self.W = torch.matmul(collection_inverse.to(self.device),
                              self.labels_bin.to(self.device))
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(self.W.T)

    def infer(self, query_features):
        with torch.no_grad():
            b, n, h, w = query_features.shape
            query_features = flatten_feature_map(query_features)
            query_features = l2_norm(query_features)[0]
            predictions = self.forward(query_features.float())
            predictions = predictions.reshape(b, h, w).squeeze(0)
        return predictions

    def forward(self, embedding):
        out = self.linear(l2_norm(embedding))
        out = torch.nn.functional.softmax(out, dim=-1)

        # apply adaptive threshold
        self.threshold = out[:, 1].max()-0.05 if self.threshold > out[:, 1].max() else self.threshold
        out = torch.where(out >= self.threshold, 1, 0)

        # get indexes of maximums
        out = out.argmax(dim=-1)
        return out
