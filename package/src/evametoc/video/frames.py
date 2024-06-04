# -*- coding: utf-8 -*-
"""This module provides routines to convert xarrays and numpy-arrays to video-frames"""


import numpy as np
import xarray as xr
import evametoc.data.nans


def from_xarrays(xarray_b, xarray_g, xarray_r, data_source="", shared_coords=False, **kwargs):
    """Writes three xarray.DataArrays to numpy arrays for each frame, and the metadata.

    Args:
        xarray_b (xarray.DataArray): A data-array with dimensions [time,width,height], for the blue channel
        xarray_g (xarray.DataArray): A data-array with dimensions [time,width,height], for the green channel
        xarray_r (xarray.DataArray): A data-array with dimensions [time,width,height], for the red channel
        data_source (str): source of the data, to be included in the metadata
        shared_coords (bool): save unique coordinates only once, not with every array
        **kwargs (dict): keyword arguments to be passed to array_to_video_frames
    Returns:
        video_frames (numpy.ndarray dtype=[uint8|uint16]): A numpy array containing the encoded data as color-values,
            to be written to a video file
        metadata (dict): metadata to be used for decoding the data, and storing coordinates
    """
    # Check function parameters
    assert len(xarray_b.shape) == 3, "All arrays must have max 3 dimensions (B)"
    assert len(xarray_g.shape) == 3, "All arrays must have max 3 dimensions (G)"
    assert len(xarray_r.shape) == 3, "All arrays must have max 3 dimensions (R)"
    assert xarray_b.shape == xarray_r.shape, "All arrays must have the same shape (B!=G)"
    assert xarray_b.shape == xarray_r.shape, "All arrays must have the same shape (B!=R)"

    data = np.concatenate([
        xarray_b.values[np.newaxis, :, :, :],
        xarray_g.values[np.newaxis, :, :, :],
        xarray_r.values[np.newaxis, :, :, :],
    ])
    if data_source:
        xarray_b.attrs["source"] = data_source
        xarray_g.attrs["source"] = data_source
        xarray_r.attrs["source"] = data_source
    metadata = {
        "B": {"name": xarray_b.name, "dims": xarray_b.dims, "attrs": xarray_b.attrs},
        "G": {"name": xarray_g.name, "dims": xarray_g.dims, "attrs": xarray_g.attrs},
        "R": {"name": xarray_r.name, "dims": xarray_r.dims, "attrs": xarray_r.attrs},
    }

    coords_b = {coord: {'values': np.atleast_1d(xarray_b.coords[coord].values), "attrs": xarray_b.coords[coord].attrs}
                for coord in xarray_b.coords.keys()}
    coords_g = {coord: {'values': np.atleast_1d(xarray_b.coords[coord].values), "attrs": xarray_b.coords[coord].attrs}
                for coord in xarray_g.coords.keys()}
    coords_r = {coord: {'values': np.atleast_1d(xarray_b.coords[coord].values), "attrs": xarray_b.coords[coord].attrs}
                for coord in xarray_r.coords.keys()}
    if shared_coords:
        coords = coords_b
        coords.update(coords_g)
        coords.update(coords_r)
        metadata["_shared_coords_"] = coords
        metadata["B"]["coord_names"] = list(xarray_b.coords.keys())
        metadata["G"]["coord_names"] = list(xarray_g.coords.keys())
        metadata["R"]["coord_names"] = list(xarray_r.coords.keys())
    else:
        metadata["B"]["coords"] = coords_b
        metadata["G"]["coords"] = coords_g
        metadata["R"]["coords"] = coords_r

    return from_numpy(data, metadata=metadata, **kwargs)


def from_numpy(data, metadata=None, bitdepth=8, nan_lossless=True):
    """Writes three 3D-numpy arrays to a video file.

    Args:
        data (np.array): Numpy array to be encoded into a video file. Dimensions: channels==3,time,width,length
        metadata (dict): data to be included into the metadata file.
            Keys must be B, G, or R for the blue (0), green (1) and red (r) channels
        bitdepth (int): The number of bits to be used to encode each color. Must be 8, 10 or 16.
        nan_lossless (bool): Encode the NaN values lossless in the metadata,
            if False, nans will be encoded as 255, which may render unexpected results
    Returns:
        video_frames (numpy.ndarray dtype=[uint8|uint16]): A numpy array containing the encoded data as color-values,
            to be written to a video file
        metadata (dict): metadata to be used for decoding the data, and storing coordinates
    """

    metadata = {"B": {}, "G": {}, "R": {}} if metadata is None else metadata

    # Check function input
    assert data.ndim == 4, "The data has to have 4 dimensions [color,time,lon,lat]"
    assert data.shape[0] == 3, "The data must have 3 colors"
    assert bitdepth in [8, 10, 16], "Only a bitdepth of 8, 10 or 16 is supported"
    assert all(k in metadata.keys() for k in "BGR"), "Metadata must have keys B, G or R"

    # Prepare the frames
    frames = np.zeros(data.shape, dtype=np.uint8 if bitdepth == 8 else np.uint16)
    max_color_value = 2 ** bitdepth - 1
    for color in range(3):
        channel_name = "BGR"[color]
        channel_data = data[color, ...]
        channel_min, channel_max = np.nanmin(channel_data), np.nanmax(channel_data)
        nan_object = {}

        if np.isnan(channel_data).any():
            if nan_lossless:
                nan_object = evametoc.data.nans.encode_nan(channel_data)
                channel_data_f = evametoc.data.nans.fill_nans(channel_data)
                channel_min, channel_max = np.nanmin(channel_data_f), np.nanmax(channel_data_f)
                channel_scaled = (channel_data_f - channel_min) / (channel_max - channel_min) * max_color_value
                channel = np.nan_to_num(channel_scaled, posinf=max_color_value, neginf=0)
            else:
                # Scale the data to 0..254 and nan to 255
                channel_scaled = (channel_data - channel_min) / (channel_max - channel_min) * (max_color_value - 1)
                channel = np.nan_to_num(channel_scaled, nan=max_color_value, posinf=max_color_value - 1, neginf=0)
                nan_object = max_color_value
        else:
            # No NaNs so, scale the data to 0..255
            channel_scaled = (channel_data - channel_min) / (channel_max - channel_min) * max_color_value
            channel = np.nan_to_num(channel_scaled, posinf=max_color_value, neginf=0)
            nan_object = -1
        frames[color, :, :, :] = channel.astype('uint8' if bitdepth == 8 else 'uint16')
        metadata[channel_name].update({"min": channel_min,
                                       "max": channel_max,
                                       "nan": nan_object,
                                       "bitdepth": bitdepth})
    return frames, metadata


def to_numpy(video_frames, metadata, metaspec=None):
    """Reads a video to a numpy-array

    Args:
        video_frames (numpy.ndarray dtype[uint8|uint16]): pixel color values from a video file
        metadata (dict): metadata of the video file
        metaspec (dict): global metadata of the video files in this folder
    Returns:
        data (numpy.array): data in the video file, descaled using the metadata.
            Dimensions [3 channels, time, width, height]
        metadata (dict): metadata of the video file
        metaspec (dict): global metadata of the video files in this folder
    """
    # Check function inputs
    assert video_frames.ndim == 4, "The data has to have 4 dimensions [color,time,lon,lat]"
    assert video_frames.shape[0] == 3, "The data must have 3 colors"
    for ch in "BGR":
        assert ["min" in metadata[ch] for ch in "BGR"], \
            f"The metadata for channel {ch:s} does not a minimum value. Cannot convert to numpy.ndarray"
        assert ["max" in metadata[ch] for ch in "BGR"], \
            f"The metadata for channel {ch:s} does not a maximum value. Cannot convert to numpy.ndarray"

    # Scale the data back to the original range
    data_arrays = []
    for color in range(3):
        channel_name = "BGR"[color]
        channel = video_frames[color, ...]

        bitdepth = metadata[channel_name].get('bitdepth', 8)
        max_color_value = 2 ** bitdepth - 1
        channel_min, channel_max = float(metadata[channel_name]['min']), float(metadata[channel_name]['max'])
        channel_data = channel

        if isinstance(metadata[channel_name]['nan'], (int, float)):
            if int(metadata[channel_name]['nan']) == -1:
                # Scale the data back from color values [0..255] to data points
                channel_data = (channel / max_color_value) * (channel_max - channel_min) + channel_min
            else:
                # Scale the data back from color values [0..254] to data points, 255 to nan
                channel_data = (channel / (max_color_value - 1)) * (channel_max - channel_min) + channel_min
                channel_data[channel == metadata[channel_name]['nan']] = np.nan
        elif 'compression' in metadata[channel_name]['nan']:
            channel_data = (channel / max_color_value) * (channel_max - channel_min) + channel_min
            nan_array = evametoc.data.nans.decode_nan(metadata[channel_name]['nan'])
            channel_data[nan_array] = np.nan
        data_arrays.append(channel_data[np.newaxis, :, :, :])
    data = np.concatenate(data_arrays)

    return data, metadata, metaspec


def to_xarray(video_frames, metadata, metaspec=None):
    """Reads a video to three xarray-DataArrays

    Args:
        video_frames (numpy.ndarray dtype[uint8|uint16]): pixel color values from a video file
        metadata (dict): metadata of the video file
        metaspec (dict): global metadata of the video files in this folder
    Returns:
        data (list[xarray.DataArray]): three DataArrays from the video file, descaled using the metadata,
            combined with coordinates and attributes from the metadata
    """
    # Check function inputs
    for ch in "BGR":
        assert ("coords" in metadata[ch] or "coord_names" in metadata[ch]), \
            f"The metadata for channel {ch:s} does not contain coordinates. Cannot convert to xarray.DataArray"
        assert "dims" in metadata[ch], \
            f"The metadata for channel {ch:s} does not contain dimensions. Cannot convert to xarray.DataArray"
        for coord_name in metadata[ch].get("coord_names", []):
            assert coord_name in metaspec["_shared_coords_"], \
                f"The metadata for channel {ch:s} references coordinate '{coord_name:s}' that are not included."

    # Frames to arrays
    data, metadata, metaspec = to_numpy(video_frames, metadata, metaspec)

    xarrays = []
    for color in range(3):
        channel_name = "BGR"[color]
        channel_attrs = metadata[channel_name].get("attrs", {})
        channel_coords = {}
        channel_coords_attrs = {}

        # First add the dims, to make sure they are in the right order
        for coord_name in metadata[channel_name]["dims"]:
            if coord_name in metadata[channel_name].get("coords", {}):
                coord_object = metadata[channel_name]["coords"][coord_name]
            else:
                coord_object = metaspec["_shared_coords_"][coord_name]
            if isinstance(coord_object, dict):
                coord_values = coord_object.get("values", [])
                channel_coords_attrs[coord_name] = coord_object.get("attrs", {})
            else:
                coord_values = coord_object
            channel_coords[coord_name] = coord_values

        # Coordinates without dimension, or length=1, cannot be added to a xarray.DataArray object
        for coord_name in metadata[channel_name].get("coords", {}).keys():
            if coord_name not in channel_coords:
                coord_object = metadata[channel_name]["coords"][coord_name]
                if isinstance(coord_object, dict):
                    channel_attrs[coord_name] = coord_object.get("values", [])
                else:
                    channel_attrs[coord_name] = coord_object
        for coord_name in metadata[channel_name].get("coord_names", []):
            if coord_name not in channel_coords:
                coord_object = metaspec["_shared_coords_"][coord_name]
                if isinstance(coord_object, dict):
                    channel_attrs[coord_name] = coord_object.get("values", [])
                else:
                    channel_attrs[coord_name] = coord_object

        array_name = metadata[channel_name].get("name", channel_name)
        xarray_obj = xr.DataArray(
            data[color, ...],
            name=array_name,
            attrs=channel_attrs,
            coords=channel_coords)
        for coord_name, coord_attrs in channel_coords_attrs.items():
            xarray_obj.coords[coord_name].attrs = coord_attrs
        xarrays.append(xarray_obj)
    return xarrays
