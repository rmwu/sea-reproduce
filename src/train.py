"""
Main training file. Call via train.sh
"""
import os
import sys
import yaml
import time
import random
from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from args import parse_args
from data import DataModule
from model import load_model
from utils import printt, get_suffix, save_pickle


# TODO why not?
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    printt("Starting...")
    with open("data/goodluck.txt") as f:
        for line in f:
            print(line, end="")

    args = parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    # save args (do not pickle object for readability)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # NOTE: does not capture change to save_path args after wandb init
    with open(args.args_file, "w+") as f:
        yaml.dump(args.__dict__, f)

    # setup
    set_seed(args.seed)
    model = load_model(args)
    printt("Finished loading model.")

    # logger
    if args.debug:
        wandb_logger = None
    else:
        name = str(time.time())
        wandb_logger = WandbLogger(project=args.run_name,
                                   name=name)
        wandb_logger.watch(model)  # gradients
        args.save_path = os.path.join(args.save_path, name)

    # train loop
    mode = "max"
    for keyword in ["loss"]:
        if keyword in args.metric:
            mode = "min"

    checkpoint_kwargs = {
        "save_top_k": 1,
        "monitor": args.metric,
        "mode": mode,
        "filename": get_suffix(args.metric),
        "dirpath": args.save_path,
        "save_last": True,
    }
    # checkpoint_path is a PTH to resume training
    #if os.path.exists(args.checkpoint_path):
    #    checkpoint_kwargs["dirpath"] = args.checkpoint_path
    cb_checkpoint = ModelCheckpoint(**checkpoint_kwargs)

    cb_earlystop = EarlyStopping(
            monitor=args.metric,
            patience=args.patience,
            mode=mode,
    )
    cb_lr = LearningRateMonitor(
            logging_interval="step"
    )
    callbacks=[
            RichProgressBar(),
            cb_checkpoint,
            cb_earlystop,
            #cb_lr
    ]
    if args.no_tqdm:
        callbacks[0].disable()

    device_ids = [args.gpu + i for i in range(args.num_gpu)]

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "accumulate_grad_batches": args.accumulate_batches,
        "gradient_clip_val": 1.,
        # evaluate more frequently
        "limit_train_batches": 200,
        "limit_val_batches": 50,
        # logging and saving
        "callbacks": callbacks,
        "log_every_n_steps": args.log_frequency,
        "fast_dev_run": args.debug,
        "logger": wandb_logger,
        # GPU utilization
        "devices": device_ids,
        "accelerator": "gpu",
        "strategy": "ddp"
        #"precision": 16,  # doesn't work well with gies?
    }

    trainer = pl.Trainer(**trainer_kwargs)
    printt("Initialized trainer.")

    # data loaders
    data = DataModule(args)
    printt("Finished loading raw data.")

    # if applicable, restore full training
    fit_kwargs = {}
    if os.path.exists(args.checkpoint_path):
        fit_kwargs["ckpt_path"] = args.checkpoint_path
    trainer.fit(model, data, **fit_kwargs)

    # FREEZE MODEL
    model.eval()

    # NOTE: I usually run inference.sh instead of using outputs here,
    # for more configuration flexibility.
    # output is also slightly different, FYI.
    if not args.debug:
        best_path = cb_checkpoint.best_model_path
        printt(best_path)
    else:
        best_path = None

    # run inference on test split
    tester = pl.Trainer(devices=[args.gpu],
                        num_nodes=1,
                        enable_checkpointing=False,
                        logger=False,
                        accelerator="gpu")

    model = model.load_from_checkpoint(best_path)
    results = tester.predict(model, data)
    # post-process results for dispatcher
    results_dict = defaultdict(list)
    for batch in results:
        for k, v in batch.items():
            if type(v) is list:
                results_dict[k].extend(v)
            else:
                results_dict[k].append(v)
    # organize by data setting
    key_to_metrics = defaultdict(list)
    auc = results_dict["auroc"]
    prc = results_dict["auprc"]
    acc = results_dict["acc"]
    for i, key in enumerate(results_dict["key"]):
        key_to_metrics[f"{key}_auc"].append(auc[i])
        key_to_metrics[f"{key}_prc"].append(prc[i])
        key_to_metrics[f"{key}_acc"].append(acc[i])
    for k, v in key_to_metrics.items():
        results_dict[k] = np.mean(v).item()
    save_pickle(args.results_file, results_dict)
    printt("All done. Exiting.")


if __name__ == "__main__":
    main()

