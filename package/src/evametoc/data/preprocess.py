# -*- coding: utf-8 -*-
"""This module provides some preprocessing functions, that need to be applied before conversion to a video file

Video files only support 3-dimensions: Width, Height and Time. Therefore, the input data needs some preprocessing.
"""
import numpy as np
import xarray as xr


def split_levels(dataset_with_levels):
    """Splits levels-dimension in separate arrays, if a levels dimension exists
    Can be undone by postprocess.merge_levels

    Args:
        dataset_with_levels (xarray.Dataset): A xarray.dataset with dimensions [time,levels,width,height]
    Returns:
        dataset (xarray.Dataset): A xarray.dataset with dimensions [time,width,height]
    """
    data_xarrays = {}
    for param_name in dataset_with_levels.keys():
        data_xarray = dataset_with_levels[param_name]
        if dataset_with_levels[param_name].ndim not in [3, 4]:
            print(f'Skipping {param_name}, has less than 3 dimensions or more than 4')
            continue
        elif dataset_with_levels[param_name].ndim == 3:
            # No levels, nothing to do
            data_xarrays[param_name] = data_xarray
        elif dataset_with_levels[param_name].ndim == 4:
            # For each level create array and save to data_xarrays
            level_dim_name = data_xarray.dims[1]
            for levelidx in range(data_xarray.shape[1]):
                flat_data_xarray = data_xarray[:, levelidx, ...]
                flat_data_xarray = flat_data_xarray.squeeze(drop=True).reset_coords(drop=True)
                level_values = dataset_with_levels.coords[level_dim_name][levelidx].values
                flat_data_xarray.attrs[f'_z_{level_dim_name:s}'] = level_values
                flat_param_name = f'{param_name:s}|z{levelidx:03d}' if data_xarray.shape[1] > 1 else param_name
                data_xarrays[flat_param_name] = flat_data_xarray
    return xr.Dataset(data_xarrays)


def remove_no_data_params(dataset, metadata=None):
    """Removes parameters that do not contain any data, or have a constant value
    Can be undone by postprocess.readd_no_data_params

    Args:
        dataset (xarray.Dataset): An dataset that may contain parameters with a constant value
        metadata (dict): A dictionary containing other metadata that should be kept.
    Returns:
        dataset (xarray.Dataset): An dataset with those parameters removed
        metadata (dict): A dictionary that may be used to rebuild those parameters later on
    """
    ds = dataset.copy()
    if metadata is None:
        metadata = {'_no_data_vars_': {}}
    else:
        metadata = metadata.copy()
    if '_no_data_vars_' not in metadata:
        metadata['_no_data_vars_'] = {}

    for param_name in ds.keys():
        data_xarray = ds[param_name]
        if np.isnan(data_xarray.values).all():
            metadata['_no_data_vars_'][param_name] = {'value': 'nan', 'dims': data_xarray.dims,
                                                      'shape': data_xarray.values.shape, 'attrs': data_xarray.attrs}
            ds = ds.drop_vars([param_name])
        elif np.nanmax(data_xarray.values) - np.nanmin(data_xarray.values) == 0:
            metadata['_no_data_vars_'][param_name] = {'value': np.nanmax(data_xarray.values), 'dims': data_xarray.dims,
                                                      'shape': data_xarray.values.shape, 'attrs': data_xarray.attrs}
            ds = ds.drop_vars([param_name])
    return ds, metadata


def make_num_params_multi3(dataset):
    """Since videos have 3 color channels, we need multitudes of 3 parameters. This function appends dummy parameters to
    achieve that.

    Args:
        dataset (xarray.Dataset): An dataset that may contain any number of parameters
    Returns:
        dataset (xarray.Dataset): An dataset where len(parameters) % 3 == 0
    """
    ds = dataset.copy()
    param_names = list(ds.keys())
    if len(param_names) % 3 == 2:
        ds = ds.assign(__dummy1__=(ds[param_names[0]].dims, ds[param_names[0]].values, ds[param_names[0]].attrs))
    elif len(param_names) % 3 == 1:
        ds = ds.assign(__dummy1__=(ds[param_names[0]].dims, ds[param_names[0]].values, ds[param_names[0]].attrs),
                       __dummy2__=(ds[param_names[1]].dims, ds[param_names[1]].values, ds[param_names[1]].attrs))
    return ds
