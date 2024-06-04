# -*- coding: utf-8 -*-
"""This module provides some postprocessing functions, that need to be applied after data has been extracted
from a video file.

Reverse functions from preprocess.py
Video files only support 3-dimensions: Width, Height and Time, whereas the original NetCDF could have had 4 or more
dimensions. Therefore, the output data needs some postprocessing.
"""

import numpy as np
import evametoc.utils


def merge_levels(dataset):
    """Merges levels, were they were separated before across multiple vars in the dataset
    Undoes what preprocess.split_levels does

    Variables containing "|z" will be merged. The name of the dimension will be retrieved from the
    attribute starting with "_z_", as will the value within that dimension.

    Args:
        dataset (xarray.Dataset): A xarray.dataset with dimensions [time,levels,width,height]
    Returns:
        dataset (xarray.Dataset): A xarray.dataset with dimensions [time,width,height]
    """
    result = dataset.copy()
    restored_leveled_data = {}
    restored_leveled_dims = {}
    vars_to_be_dropped = []

    for k in list(result.keys()):
        if '|z' in k:
            # Parameter name contains |z, so it is a level from a separated variable
            # Get the name of the level dimension, its value within that dimension,
            # name of the original var and level-index
            data_array = result[k]
            param_name, level = k.split('|z')
            level = int(level)
            z_dim_name, z_dim_value = 'z', level
            for attr in data_array.attrs.keys():
                if attr.startswith('_z_'):
                    z_dim_name, z_dim_value = attr[3:], data_array.attrs[attr]
                    data_array.attrs.pop(attr)
                    break

            # Store the level-index/value, so we can rebuild this dimension later on
            if z_dim_name not in restored_leveled_dims:
                restored_leveled_dims[z_dim_name] = {}
            restored_leveled_dims[z_dim_name][level] = z_dim_value

            # Store the dimensions, data and attributes, so we can rebuild the variable
            if param_name not in restored_leveled_data:
                dims = [data_array.dims[0], z_dim_name] + list(data_array.dims[1:])
                restored_leveled_data[param_name] = {'dims': dims, 'levels': {}, 'attrs': {}}
            restored_leveled_data[param_name]['attrs'].update(data_array.attrs)
            restored_leveled_data[param_name]['levels'][level] = data_array.values[:, np.newaxis, :, :]

            # Schedule the separated var to be deleted
            vars_to_be_dropped.append(k)

    # Order the dimensions by level-index, and alphabetically
    for k in sorted(restored_leveled_dims):
        restored_leveled_dims[k] = list(evametoc.utils.sort_dict_by_keys(restored_leveled_dims[k]).values())
    result = result.assign_coords(**restored_leveled_dims)

    # Add the new merged variables
    for param_name in sorted(restored_leveled_data):
        data = []
        for level in sorted(restored_leveled_data[param_name]['levels']):
            data.append(restored_leveled_data[param_name]['levels'][level])
        data = np.concatenate(data, axis=1)
        result = result.assign(**{param_name: (
            restored_leveled_data[param_name]['dims'],
            data,
            restored_leveled_data[param_name]['attrs']
        )})
    # delete the old ones
    result = result.drop_vars(vars_to_be_dropped)

    return result


def readd_no_data_params(dataset, metaspec):
    """Readds parameters that were removed for not containing any data, or have a constant value
    Undoes what preprocess.remove_no_data_params does

    Args:
        dataset (xarray.Dataset): The dataset with parameters removed
        metaspec (dict): The global metadata (such as _shared_coords_) present in the metadata-file
    Returns:
        dataset (xarray.Dataset): A copy of dataset with those parameters readded
    """
    if '_no_data_vars_' not in metaspec:
        return dataset
    if not metaspec['_no_data_vars_']:
        return dataset
    
    ds = dataset.copy()

    first_xarray = ds[list(ds.keys())[0]]
    for param_name, properties in metaspec['_no_data_vars_'].items():
        value = properties.get('value', 'nan')
        dims = properties.get('dims', first_xarray.dims)
        shape = properties.get('shape', first_xarray.values.shape)
        attrs = properties.get('attrs', {})
        if value == 'nan':
            value = np.nan
        ds = ds.assign(**{param_name: (dims, np.full(shape, value), attrs)})
    return ds
