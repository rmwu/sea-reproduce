"""
DataModule is entrypoint into all data-related objects.

We create different datasets via diamond inheritence
which is arguably horrible and clever hehe.

"""

import os
from functools import partial
from contextlib import redirect_stdout

import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from .dataset import MetaObservationalDataset, MetaInterventionalDataset
from .dataset import TrainDataset, TestDataset, BaselineDataset
from .utils import collate


def get_base_dataset(algorithm):
    if algorithm in ["fci", "ges", "grasp"]:
        return MetaObservationalDataset
    elif algorithm in ["gies"]:
        return MetaInterventionalDataset
    else:
        raise Exception("Unsupported algorithm", algorithm)


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_file = args.data_file

        BaseDataset = get_base_dataset(args.algorithm)
        class BaseTrainDataset(BaseDataset, TrainDataset):
            pass
        class BaseTestDataset(BaseDataset, TestDataset):
            pass

        self.subset_train = BaseTrainDataset(self.data_file, args,
                splits_to_load=["train"])
        self.subset_val = BaseTestDataset(self.data_file, args,
                splits_to_load=["val"])
        self.subset_test = BaseTestDataset(self.data_file, args,
                splits_to_load=["test"])

    def train_dataloader(self):
        train_loader = DataLoader(self.subset_train,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  persistent_workers=(not self.args.debug),
                                  collate_fn=partial(collate, self.args))
        return train_loader

    def val_dataloader(self):
        # batch_size smaller since we sample more batches on average
        val_loader = DataLoader(self.subset_val,
                                batch_size=max(self.batch_size // 4, 1),
                                num_workers=max(self.num_workers // 4, 1),
                                shuffle=False,
                                pin_memory=True,
                                persistent_workers=(not self.args.debug),
                                collate_fn=partial(collate, self.args))
        return val_loader

    def predict_dataloader(self):
        test_loader = DataLoader(self.subset_test,
                                 batch_size=max(self.batch_size // 4, 1),
                                 num_workers=self.num_workers,
                                 shuffle=False,
                                 pin_memory=False,
                                 collate_fn=partial(collate, self.args))
        return test_loader

    def test_dataloader(self):
        return self.predict_dataloader()


class InferenceDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        # for proper timing, set batch_size to 1
        self.batch_size = 1
        self.num_workers = args.num_workers
        self.data_file = args.data_file

        BaseDataset = get_base_dataset(args.algorithm)
        class BaseTestDataset(BaseDataset, TestDataset):
            pass

        self.subset_test = BaseTestDataset(self.data_file, args,
                splits_to_load=["test"])

    def predict_dataloader(self):
        test_loader = DataLoader(self.subset_test,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=False,
                                 pin_memory=False,
                                 collate_fn=partial(collate, self.args))
        return test_loader

    def test_dataloader(self):
        return self.predict_dataloader()


class BaselineDataModule(pl.LightningDataModule):
    """
        Used for running baseline algorithms only. Samples all variables.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_file = args.data_file

        BaseDataset = get_base_dataset(args.algorithm)
        class BaseTestDataset(BaseDataset, BaselineDataset):
            pass

        self.subset_test = BaseTestDataset(self.data_file, args,
                splits_to_load=["test"])

    def predict_dataloader(self):
        test_loader = DataLoader(self.subset_test,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=False,
                                 pin_memory=False,
                                 collate_fn=partial(collate, self.args))
        return test_loader

    def test_dataloader(self):
        return self.predict_dataloader()

