# -*- coding: utf-8 -*-
"""This module provides grouping functions, that group the parameters in sets of 3, to be encoded into one video."""


import random


def by_unit(dataset):
    """Creates groups of three variables to be stored in one video (Blue,Green,Red), based on their unit

    Args:
        dataset (xarray.Dataset): A dataset containing all variables
    Returns:
        groups (list<list<3 x string>>): List of groups of 3 keys that could be stored together in one video file
    """
    params = list(dataset.keys())
    if len(params) % 3 == 1:
        print("Number of fields in the dataset are not dividable by three!")
        print("Dropping:", params.pop())
    elif len(params) % 3 == 2:
        print("Number of fields in the dataset are not dividable by three!")
        print("Dropping:", params.pop(), params.pop())
    units = {param: dataset[param].attrs.get('units', '-') for param in params}
    param_levels = {param: param.split('|') for param in params}
    params_sorted = sorted(params,
                           key=lambda param_name: (units[param_name], *param_levels[param_name]))
    params_grouped = list(zip(params_sorted[0::3], params_sorted[1::3], params_sorted[2::3]))
    return params_grouped


def group_random(dataset):
    """Creates groups of three variables to be stored in one video (Blue,Green,Red), randomly

    Args:
        dataset (xarray.Dataset): A dataset containing all variables
    Returns:
        groups (list<list<3 x string>>): List of groups of 3 keys that could be stored together in one video file
    """
    params = list(dataset.keys())
    if len(params) % 3 == 1:
        print("Number of fields in the dataset are not dividable by three!")
        print("Dropping:", params.pop())
    elif len(params) % 3 == 2:
        print("Number of fields in the dataset are not dividable by three!")
        print("Dropping:", params.pop(), params.pop())
    random.shuffle(params)
    params_grouped = list(zip(params[0::3], params[1::3], params[2::3]))
    return params_grouped
