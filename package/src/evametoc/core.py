# -*- coding: utf-8 -*-
"""This module provides the main routines for encoding and decoding videos to environmental arrays."""
import glob
import os
import tempfile
import xarray as xr
import numpy as np

import evametoc.data
import evametoc.evafile
import evametoc.metadata
import evametoc.video
import evametoc.utils


def main_encode(dataset_path, file_size, target_path=None, dataset_type=None, nan_lossless=True, **kwargs):
    """Encodes a dataset to a .eva video-fileset

    Args:
        dataset_path (str): Path to a NetCDF dataset
        file_size (int): Number of bytes to be used in the final file size (approx)
        target_path (str|None): Destination path to the eva file
        dataset_type (str): One of ['harm40','GFS','copernicus_ocean',None] denoting the type of dataset
        nan_lossless (bool): Encode the NaNs losslessly in the metadata
        kwargs (dict): to be passed along to video_encoder_ffmpeg
    """
    # Check function arguments
    assert os.path.isfile(dataset_path), f"File '{dataset_path}' does not exist"
    assert dataset_type in ['harm40', 'GFS', 'copernicus_ocean',
                            None], f"Dataset-type '{dataset_type}' is not supported"

    # Configure function arguments
    target_path = evametoc.utils.find_target_path(dataset_path, '.eva', target_path, overwrite=True)
    video_file_ext = kwargs.pop('video_file_ext', 'mp4')

    # Init
    global_metadata = {}

    # Main process
    ds = xr.open_dataset(dataset_path)
    global_metadata['_dataset_attrs_'] = ds.attrs.copy()
    global_metadata['_eva_compression_'] = evametoc.utils.human_readable_file_size(file_size)

    expected_nan_size_lossless = np.mean(np.array([ds[k][0, ...].size for k in ds.keys()]))/8/1.5
    
    ds = evametoc.data.prepare.prepare(ds, dataset_type)
    ds = evametoc.data.preprocess.split_levels(ds)
    ds, global_metadata = evametoc.data.preprocess.remove_no_data_params(ds, global_metadata)
    ds = evametoc.data.preprocess.make_num_params_multi3(ds)

    groups = evametoc.data.group.by_unit(ds)
    if file_size is None or file_size <= 0:
        file_size_per_video = None
    else:
        file_size_per_video = file_size / len(groups)
    
    with tempfile.TemporaryDirectory(prefix='.evac_', dir='.') as tmp_dir_name:
        for group_id, group in enumerate(groups):
            frames, metadata = evametoc.video.frames.from_xarrays(
                ds[group[0]],
                ds[group[1]],
                ds[group[2]],
                shared_coords=True,
                nan_lossless=nan_lossless)

            video_file_path = os.path.join(tmp_dir_name, f'video_{group_id:04d}.{video_file_ext:s}')
            if group_id == 0:
                metadata.update(global_metadata)
            video_file_path, metadata = evametoc.video.ffmpeg.encode(
                video_file=video_file_path,
                video_frames=frames,
                metadata=metadata,
                file_size=file_size_per_video,
                **kwargs)
            evametoc.metadata.write(video_file_path, metadata)
        evametoc.evafile.dir_to_eva(tmp_dir_name, target_path)


def main_decode(eva_path, target_path=None, overwrite=False, **kwargs):
    """Decodes a dataset from a .eva video-fileset

    Args:
        eva_path (str): Path to an environmental video array (.eva file)
        target_path (str|None): Destination path to store the dataset/.nc-file
        overwrite (bool): Overwrite an existing file, if already present
        kwargs (dict): Keyword arguments passed along to video_decoder_ffmpeg
    """
    # Check function arguments
    assert os.path.isfile(eva_path), f"File '{eva_path}' does not exist"

    # Configure function arguments
    target_path = evametoc.utils.find_target_path(eva_path, '.eva_transfer.nc', target_path, overwrite=overwrite)
    if not overwrite:
        assert not os.path.isfile(target_path), f"File '{target_path}' does already exist"

    with tempfile.TemporaryDirectory(prefix='.evax_', dir='.') as tmp_dir_name:
        evametoc.evafile.eva_to_dir(eva_path, tmp_dir_name)

        arrays = {}
        for video_file in glob.iglob(os.path.join(tmp_dir_name, '*.*')):
            if video_file.endswith('.json'):
                continue
            metadata, metaspec = evametoc.metadata.read(video_file)
            video_frames, metadata, metaspec = evametoc.video.ffmpeg.decode(video_file, metadata, metaspec, **kwargs)

            xarrays = evametoc.video.frames.to_xarray(video_frames, metadata, metaspec)

            for i in range(len(xarrays)):
                if '__dummy' in xarrays[i].name:
                    continue
                arrays[xarrays[i].name] = xarrays[i]

        dataset = xr.Dataset(arrays)
        dataset.attrs['EVA'] = (
            'Decompressed data from an {} environmental video array'.format(metaspec.get('_eva_compression_', '?B')))
        dataset.attrs.update(metaspec['_dataset_attrs_'])

        dataset = evametoc.data.postprocess.readd_no_data_params(dataset, metaspec)
        dataset = evametoc.data.postprocess.merge_levels(dataset)

        comp = {'zlib': True, 'complevel': 5, 'least_significant_digit': 3}
        encoding = {var: comp for var in dataset.data_vars}
        dataset.to_netcdf(target_path, format='NETCDF4', encoding=encoding)
