from __future__ import absolute_import
from .duke import Duke
from .market import Market
from .cuhk import CUHK
from .msmt import MSMT
from .randperson import RandPerson

__factory = {
    'market': Market,
    'market1501': Market,
    'dukemtmc': Duke,
    'cuhk03_np_labeled': CUHK,
    'cuhk03_np_detected': CUHK,
    'cuhk03_np': CUHK,
    'msmt': MSMT,
    'msmt17': MSMT,
    'randperson': RandPerson,
    'randperson_subset': RandPerson
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'market', 'duke'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
