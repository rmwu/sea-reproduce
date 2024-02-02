import os
import sys

from .aggregator import Aggregator
from .baseline import CausalBaseline


def get_model_cls(args):
    """
        Get class only
    """
    if args.model == "baseline":
        cls = CausalBaseline
    elif args.model == "aggregator":
        cls = Aggregator
    else:
        raise Exception(f"Invalid model {args.model}")
    return cls


def load_model(args, data_module=None, **kwargs):
    """
        Model factory
    """
    if args.model == "baseline":
        model = CausalBaseline(args, **kwargs)
    elif args.model == "aggregator":
        model = Aggregator(args, **kwargs)
    else:
        raise Exception(f"Invalid model {args.model}")

    return model

