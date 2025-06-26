import torch

from collections import OrderedDict, namedtuple



def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


def is_list(x):
    return isinstance(x, list) or isinstance(x, torch.nn.ModuleList)


class NoOp:
    """A no-op class for the case when we don't want to do anything."""

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, *args, **kwargs):
        def no_op(*args, **kwargs): pass

        return no_op

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

    def __enter__(self):
        pass

    def dampening(self):
        return self

    def reloading(self):
        return self


class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v
