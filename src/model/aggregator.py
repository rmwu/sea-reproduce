"""
Sample, estimate, aggregate.

Aggregate = given local estimates, resolve discrepancies and
produce final, global graph.
"""

import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
#from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.classification import BinaryAccuracy

from .axial import AxialTransformer, TopLayer
from .utils import get_params_groups


class Aggregator(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_vars = args.num_vars

        # Transformer on sequence of predicted graphs
        self.encoder = AxialTransformer(args)
        self.top_layer = TopLayer(
            embed_dim=self.args.embed_dim * 2,
            output_dim=3
        )

        self.bce_loss = nn.BCELoss()

        # validation meters
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()
        self.acc = BinaryAccuracy()

        self.save_hyperparameters()

    def forward(self, batch):
        """
        Used on predict_dataloader
        """
        start = time.time()  # keep track of GPU time
        output = self.encoder(batch)
        results = self.compute_metrics_per_graph(output, batch,
                save_preds=True)
        end = time.time()  # keep track of GPU time
        # run with batch_size=1 for accurate timing
        results["time"] = batch["time"] + (end - start)
        return results

    def symmetrize(self, output, batch, reduce=True):
        """
            P(i->j) = 1 - P(j->i)
            reduce: bool  True to reduce batch, False to preserve graphs
        """
        # symmetrize output
        # select upper and lower triangular, skip diagonal
        # NOTE: tril is NOT same as triu of transpose
        output = output.permute(0, 3, 1, 2)  # move embed_dim to 2 for triu
        forward_edge = torch.triu(output, 1)
        backward_edge = torch.triu(output.transpose(2, 3), 1)
        forward_edge = forward_edge.permute(0, 2, 3, 1)
        backward_edge = backward_edge.permute(0, 2, 3, 1)
        # select away padding as well as wrong direction
        # backward_mask should == forward_mask
        forward_mask = forward_edge[...,0] != 0.  # remove embed_dim dimension
        backward_mask = backward_edge[...,0] != 0.
        if reduce:
            forward_edge = forward_edge[forward_mask]
            backward_edge = backward_edge[backward_mask]
            assert len(forward_edge) == len(backward_edge)
            logits = torch.cat([forward_edge, backward_edge], dim=1)
            edge_pred = self.top_layer(logits)  # (B*T, 2*dim) -> (B*T, 3)
        else:
            edge_pred = []
            for i in range(len(forward_edge)):
                forward_i = forward_edge[i][forward_mask[i]]
                backward_i = backward_edge[i][backward_mask[i]]
                logits = torch.cat([forward_i, backward_i], dim=-1)
                edge_pred.append(self.top_layer(logits))

        # get label corresponding to same entries
        label = batch["label"]
        forward_label = torch.triu(label, 1)
        backward_label = torch.triu(label.transpose(1, 2), 1)
        if reduce:
            forward_label = forward_label[forward_mask]
            backward_label = backward_label[backward_mask] * 2
            # forward/backward should be mutually exclusive
            joint_label = forward_label + backward_label  # {0, 1, 2}
        else:
            joint_label = []
            for i in range(len(forward_edge)):
                forward_i = forward_label[i][forward_mask[i]]
                backward_i = backward_label[i][backward_mask[i]]
                joint_label.append(torch.cat([forward_i, backward_i]))
        return edge_pred, joint_label

    def compute_losses(self, output, batch):
        losses = {}
        pred, true = self.symmetrize(output, batch)
        losses["loss"] = F.cross_entropy(pred, true)
        return losses

    def compute_metrics_per_graph(self, output, batch, save_preds=False):
        """
            Split up individual graphs from batch
        """
        auroc, auprc, acc = [], [], []
        if save_preds:
            pred_list, true_list = [], []
        # do not reduce over batch
        pred, true = self.symmetrize(output, batch, reduce=False)
        for i, (p, t) in enumerate(zip(pred, true)):
            # select P(forward) and P(backward), remove P(no edge)
            # corresponding to dim=1, dim=2, and dim=0
            p = F.softmax(p, dim=-1)[:,1:].t().reshape(-1)  # (T,)
            p = p.cpu()
            t = t.cpu()
            assert p.shape == t.shape
            # convert prediction to list
            auroc.append(self.auroc(p, t).item())
            auprc.append(self.auprc(p, t).item())
            acc.append(self.acc(p, t).item())
            if save_preds:
                pred_list.append(p.tolist())
                true_list.append(t.tolist())

        outputs = {}
        outputs["auroc"] = auroc
        outputs["auprc"] = auprc
        outputs["acc"] = acc
        # need to save these...
        outputs["key"] = batch["key"]
        if save_preds:
            outputs["pred"] = pred_list
            outputs["true"] = true_list
        return outputs

    def training_step(self, batch, batch_idx):
        # cuda oom
        try:
            output = self.encoder(batch)
            losses = self.compute_losses(output, batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return
        for k, v in losses.items():
            if not torch.is_tensor(v) or v.numel() == 1:
                self.log(f"Train/{k}", v.item(),
                    batch_size=len(output), sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        # cuda oom
        try:
            output = self.encoder(batch)
            losses = self.compute_losses(output, batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return
        for k, v in losses.items():
            if not torch.is_tensor(v) or v.numel() == 1:
                self.log(f"Val/{k}", v.item(),
                    batch_size=len(output), sync_dist=True)
        # metrics
        results = self.compute_metrics_per_graph(output, batch,
                save_preds=False)
        for k, v in results.items():
            if k != "key":
                self.log(f"Val/{k}", np.mean(v),
                    batch_size=len(output), sync_dist=True)

    def configure_optimizers(self):
        param_groups = get_params_groups(self, self.args)
        optimizer = AdamW(param_groups)
        # scheduler makes everything worse =__=
        #scheduler = CosineAnnealingLR(optimizer, self.args.epochs)
        return [optimizer]#, [scheduler]

