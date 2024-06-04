# -*- coding: utf-8 -*-
"""This module provides the file-type specific preparation functions for this package

Some files (like harmonie-files) contain accumulated data, nan's, stacked data or static data.
This module provides functions to handle those datasets, and prepares them for use in this package
"""

import logging

import numpy as np
import xarray as xr


def prepare(dataset, dataset_type=None):
    """Finds and applies the relevant preparation functions

    Args:
        dataset (xarray.Dataset): The dataset to be prepared.
        dataset_type (str): The type of dataset to be prepared.
            One of ['harm40','GFS','copernicus_ocean', None].
            If no valid dataset_type was selected, no preparation is applied.
    Returns:
        prepared_dataset (xarray.Dataset): The dataset with relevant preparation functions applied
    """
    prepare_functions = {
        'harm40': [drop_knmi_landsea, differentiation_knmi],
        'GFS': [],
        'copernicus_ocean': [],
        None: [],
    }

    steps = prepare_functions.get(dataset_type, None)
    if steps is None:
        logging.warning(f'The dataset_type "{dataset_type:s}" was unknown. No preparation applied.')
        return dataset
    dataset = dataset.copy()
    for step in steps:
        dataset = step(dataset)
    return dataset


def drop_knmi_landsea(dataset):
    """Drops the land-sea mask from KNMI datasets

    Args:
        dataset (xarray.Dataset): A KNMI dataset that may contain a landsea mask
    Returns:
        dataset (xarray.Dataset): The input dataset, without KNMI landsea mask
    """
    ds = dataset.copy()
    for k in ds.keys():
        table = ds[k].attrs.get('table', 0)
        pcode = ds[k].attrs.get('code', 0)
        if table in [253] and pcode in [81]:
            # Land-sea mask (81) according to KNMI Grib table (253)
            ds = ds.drop_vars([k])
    return ds


def differentiation_knmi(dataset):
    """Differentiates accumulated fields in a KNMI dataset (e.g. rain, radiation)

    Args:
        dataset (xarray.Dataset): A KNMI dataset that may contain accumulated fields
    Returns:
        dataset (xarray.Dataset): The input dataset, with the accumulated fields, differentiated
    """
    ds = dataset.copy()
    for k in ds.keys():
        table = ds[k].attrs.get('table', 0)
        pcode = ds[k].attrs.get('code', 0)
        if table in [253] and pcode in [111, 112, 117, 122, 132, 181, 184, 201]:
            # These variables are cumulated according to the KNMI Grib table (253)
            # https://www.knmidata.nl/data-services/knmi-producten-overzicht/atmosfeer-modeldata/data-product-1
            ds[k] = differentiate_array(ds[k])
            ds[k].attrs['_aggr'] = 'sum'
    return ds


def differentiate_array(accumulated_data):
    """Some arrays have summed fields (over time). This method reverses that.

    Args:
        accumulated_data (xarray.DataArray): A xarray with the first dimension time, which contains sums
            over time. (For example: accumulated rain or radiation since start of this model run).
    Returns:
        data (xarray.DataArray): A xarray containing the differences between timesteps of accumulated_data
    """
    # Essentially np.diff + a zero array in the first timestep...
    dc = np.zeros(accumulated_data.shape)
    dc[0, ...] = accumulated_data[0, ...]
    dc[1:, ...] = np.diff(accumulated_data.values, axis=0)
    return xr.DataArray(dc, coords=accumulated_data.coords, name=accumulated_data.name, attrs=accumulated_data.attrs)
