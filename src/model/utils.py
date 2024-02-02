"""
Taken from Facebook's DINO
"""

import numpy as np
import torch.nn as nn

import torch


def get_params_groups(model, args):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    groups = [{"params": regularized,
               "weight_decay": args.weight_decay,
               "lr": args.lr},
              {"params": not_regularized,
               "weight_decay": 0.,
               "lr": args.lr},
               ]
    return groups


def shd_metric(pred, true):
    """
    Calculates the structural hamming distance.
    Copied from:
    https://github.com/slachapelle/dcdi/blob/594d328eae7795785e0d1a1138945e28a4fec037/dcdi/utils/metrics.py

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    shd

    """
    diff = true - pred

    rev = (((diff + diff.t()) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn + fp + rev

