# -*- coding: utf-8 -*-
"""This module provides functions for extracting, compressing, filling, decompressing and resetting Nan values in
an environmental dataset."""

import base64
import lzma
import numpy as np
import scipy.ndimage


def encode_nan(data, algorithm='lzma'):
    """Encode the nan's in data using a lossless compression algorithm

    Args:
        data (np.array): The dataset containing NaNs
        algorithm (str): Method used for compression, choose from [None,'base64','lzma']
    Returns:
        nanobj (dict):
            - compression (str): The algorithm used
            - data (str): base64 encoded nan-positions, compressed with algorithm
            - shape (list): shape of the original array
    """
    assert algorithm in [None, 'best', 'base64', 'lzma']
    algorithm = 'best' if algorithm is None else algorithm
    algorithm = 'lzma' if algorithm == 'best' else algorithm

    nanfield = np.isnan(data)

    # Often nan-s encode lack of data due to ground level
    # and are therefore time-independent. Remove time dimension if so...
    equal_over_axis0 = np.equal.reduce(nanfield, axis=0).all()
    if equal_over_axis0:
        nanflat = nanfield[0, ...].flatten()
    else:
        nanflat = nanfield.flatten()

    # Combine every 8 booleans (NaN Yes/No) to one byte.
    nanbyte = np.packbits(nanflat).tobytes()

    # Compress the byte-string using one of the python-standard compression libs
    # (All byte-strings are encoded to base64 for better support in JSON later on...)
    compressed = {}
    if algorithm == 'base64':
        compressed['base64'] = base64.b64encode(nanbyte).decode()
    elif algorithm == 'lzma':
        nanlzma = lzma.compress(nanbyte)
        compressed['lzma'] = base64.b64encode(nanlzma).decode()

    # Return a nan-object which can be implemented in the JSON file
    return {'compression': algorithm,
            'data': compressed[algorithm],
            'shape': nanfield.shape}


def fill_nans(dataset):
    """Fill the NaNs in the dataset

    Args:
        dataset (np.array): Dataset containing nans, minimal 2 dims.
    Returns:
        filled (np.array): Dataset containg all values from `dataset`, with all nans filled
    """
    if dataset.ndim > 2:
        # This function only works for 2d-slices, so slice this dimension to 2D
        filled = []
        for timestep in range(dataset.shape[0]):
            filled.append(fill_nans(dataset[timestep, ...])[None, ...])
        return np.concatenate(filled, axis=0)
    elif dataset.ndim == 2:
        # Get the NaN positions
        nans = np.isnan(dataset)

        # For each NaN, take the nearest neighbor
        nearest_index = scipy.ndimage.distance_transform_edt(
            nans, return_distances=False, return_indices=True)
        nearest_neighbor = dataset[tuple(nearest_index)]

        # Apply a gausian blur, so that the resulting image is more smooth
        gausian_blur = scipy.ndimage.gaussian_filter(nearest_neighbor, 5)

        # Make sure no datapoints were alterated during this process
        return np.where(nans, gausian_blur, dataset)
    else:
        raise ValueError("Not enough dimensions for 2D fill")


def decode_nan(nancoded):
    """Decode a nan-object

    Args:
        nancoded (dict)
            - compression (str): Method used for compression, choose from ['base64','lzma']
            - data (str): base64 encoded nan-positions, compressed with algoritm
            - shape (list): shape of the original array
    Returns:
        nanarr (np.array[bool]): True where a NaN value was present, False otherwise
    """
    assert all([k in nancoded for k in ['compression', 'data', 'shape']])
    assert nancoded['compression'] in ['base64', 'lzma']

    # Get the data
    nanbstr = base64.b64decode(nancoded['data'])

    # Decompress using the original compression lib
    if nancoded['compression'] == 'lzma':
        nanbstr = lzma.decompress(nanbstr)

    # Create a numpy.array from the compressed bytes and go from bytes to bits.
    nanbyte = np.frombuffer(nanbstr, dtype=np.uint8)
    nanbits = np.unpackbits(nanbyte) == 1

    # Reshape to the original shape
    if nanbits.size < np.prod(nancoded['shape']):
        nanarr = reshape_bits(nanbits, nancoded['shape'][1:])
        return np.repeat(nanarr[None, ...], nancoded['shape'][0], axis=0)
    return reshape_bits(nanbits, nancoded['shape'])


def reshape_bits(bitsarray, target_shape, *args, **kwargs):
    """Reshapes a bitarray back to its original shape
    
    When bits are converted to bytes, using np.packbits(nanflat) in encode_nan,
    they are padded with upto 7 bits to generate a number of bytes. This function
    corrects for that padding, and reshapes to the right target shape by removing
    the last few bits
    
    Args:
        bitsarray (np.array[bool]): 1D array of bits
        target_shape (list[int]): shape of the target array
        *args and **kwargs: passed to np.reshape
    Returns:
        bitsarray (np.array[bool]): Array with dimensions target_shape, containing the bits
    """
    assert bitsarray.size >= np.prod(target_shape)
    assert bitsarray.size < np.prod(target_shape) + 8 # Another problem might exist
    if bitsarray.size > np.prod(target_shape):
        assert bitsarray[np.prod(target_shape):].any() == False  # Content discoverd, this is not a correct solution
        bitsarray = bitsarray[:np.prod(target_shape)]
    return np.reshape(bitsarray, target_shape, *args, **kwargs)


def set_nan(data, nanarr):
    """Sets data to np.nan where nanarr is True

    Args:
        data (np.array): Array with values
        nanarr (np.array): Boolean array, True where NaN, having the same shape as data
    Returns:
        arr (np.array):
    """
    assert data.shape == nanarr.shape
    arr = data.copy()
    arr[nanarr] = np.nan
    return arr
