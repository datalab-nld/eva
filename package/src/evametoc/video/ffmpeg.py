# -*- coding: utf-8 -*-
"""This module provides routines to convert write numpy-arrays to video files"""

import os
import numpy as np
import ffmpeg


def encode(video_file, video_frames, metadata, *,
           file_size=None, codec='libx264', fps=24,
           inkwargs=None, outkwargs=None, **kwargs):
    """Writes video frame arrays to a video file, using the ffmpeg-python module

    Args:
        video_file (str): Path to the file to be written
        video_frames (numpy.ndarray dtype=[uint8|uint16]): A numpy array containing the encoded data as color-values,
            to be written to a video file
        metadata (dict): metadata to be used for decoding the data, and storing coordinates
        file_size (int): Approximate number of bytes to be used to encode the video
        codec (str): Codec to be used (see ffmpeg --codecs)
        fps (float): frames per seconds
        inkwargs (dict): extra arguments to be passed for the input stream in the ffmpeg module
        outkwargs (dict): extra arguments to be passed for the output file in the ffmpeg module
        **kwargs (dict): same as outkwargs
    """
    # Checking the inputs
    assert video_frames.ndim == 4, "The data has to have 4 dimensions [color,time,lon,lat]"
    assert video_frames.shape[0] == 3, "The data must have 3 colors"
    assert all(k in metadata.keys() for k in "BGR"), "Metadata must have keys B, G or R"

    # Preparing the frames
    video_frames = np.transpose(video_frames, (1, 2, 3, 0))  # color as last
    timesteps, width, height, colors = video_frames.shape
    bitdepth = 8  # metadata['B'].get('bitdepth', 8)

    # Processing the input and output keywords
    inkw = {
        'f': 'rawvideo',
        'pix_fmt': 'bgr24' if bitdepth == 8 else f'gbrp{bitdepth:02d}le',
        'r': 24,
        's': '{}x{}'.format(height, width)
    }
    inkw.update(inkwargs if inkwargs is not None else {})
    outkw = {
        'c:v': codec,
        'pass': 1,
        'r': fps,
        'loglevel': 'error',
    }
    if file_size is not None:
        outkw['b:v'] = (file_size * 8) / (1.073741824 * timesteps / fps)  # = Bitrate [bit/s]
    outkw.update(outkwargs if outkwargs is not None else {})
    outkw.update(kwargs)

    # Init the video-writer
    ffmpeg_procedure = (
        ffmpeg
        .input('pipe:', **inkw)
        .output(video_file, **outkw)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # Write the frames one by one
    if 'gbrp' in inkw.get('pix_fmt', ''):
        # Green, Blue, Red Planar (Plains before next color chanel)
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))
        video_frames = np.concatenate([
            video_frames[:, [1], :, :],  # Green
            video_frames[:, [0], :, :],  # Blue
            video_frames[:, [2], :, :],  # Red
        ], axis=1)
    for ti in range(timesteps):
        ffmpeg_procedure.stdin.write(video_frames[ti, :, :, :].tobytes())

    # Close the video-writer and write the metadata
    ffmpeg_procedure.stdin.close()
    ffmpeg_procedure.wait()
    return video_file, metadata


def decode(video_file, metadata, metaspec, inkwargs=None, outkwargs=None, **kwargs):
    """Reads a video file to a numpy-array of color values, using the ffmpeg-python module

        Args:
            video_file (str): path to the video file
            metadata (dict): metadata for this video file
            metaspec (dict): global metadata (such as _shared_coords_) present in the metadata-file
            inkwargs (dict): extra arguments to be passed for the input file in the ffmpeg module
            outkwargs (dict): extra arguments to be passed for the output stream in the ffmpeg module
            **kwargs: same as inkwargs
        Returns:
            video_frames (numpy.ndarray dtype[uint8|uint16]) array with shape [colors,time,width,height] containing
                the color values for each pixel
            metadata (dict): metadata for this video file
            metaspec (dict): global metadata (such as _shared_coords_) present in the metadata-file
        """
    # Check function inputs
    assert os.path.isfile(video_file), "Video file does not exist"
    for ch in "BGR":
        assert ("coords" in metadata[ch] or "coord_names" in metadata[ch]), \
            f"The metadata for channel {ch:s} does not contain coordinates. Cannot convert to xarray.DataArray"
        assert "dims" in metadata[ch], \
            f"The metadata for channel {ch:s} does not contain dimensions. Cannot convert to xarray.DataArray"
        for coord_name in metadata[ch].get("coord_names", []):
            assert coord_name in metaspec["_shared_coords_"], \
                f"The metadata for channel {ch:s} references " \
                f"coordinate '{coord_name:s}' that are not included in the file."

    # Determine the dimension names
    time_name = metadata["B"]['dims'][0]
    width_name = metadata["B"]['dims'][1]
    height_name = metadata["B"]['dims'][2]

    coords = metadata["B"]["coords"] if "coords" in metadata["B"] else metaspec["_shared_coords_"]
    # LEGACY In previous version coord was a list of values, now a dict with values and attributes.
    timesteps = len(coords[time_name]['values'] if isinstance(coords[time_name], dict) else coords[time_name])
    width = len(coords[width_name]['values'] if isinstance(coords[width_name], dict) else coords[width_name])
    height = len(coords[height_name]['values'] if isinstance(coords[height_name], dict) else coords[height_name])
    bitdepth = metadata["B"].get('bitdepth', 8)

    # Processing the input and output keywords
    inkw = {}
    inkw.update(inkwargs if inkwargs is not None else {})
    inkw.update(kwargs)
    outkw = {
        'format': 'rawvideo',
        'pix_fmt': 'bgr24' if bitdepth == 8 else 'gbrp16le',
        'loglevel': 'error',
        'pass': 1}
    outkw.update(outkwargs if outkwargs is not None else {})

    # Init the reader
    ffmpeg_procedure = ffmpeg.input(video_file, **inkw).output('pipe:', **outkw).run_async(pipe_stdout=True)

    # Read all bytes:
    all_bytes = b''
    while True:
        read_bytes = ffmpeg_procedure.stdout.read(width * height)
        if not read_bytes:
            break
        all_bytes += read_bytes
    video_array = np.frombuffer(all_bytes, np.uint8 if bitdepth == 8 else np.uint16)
    
    if 'gbrp' in outkw.get('pix_fmt', ''):
        # gbrp16le has a slightly different order of pixels in the bitstream. Therefore, reorder
        video_frames = np.transpose(video_array.reshape(timesteps, 3, width, height), (1, 0, 2, 3))
        video_frames = np.concatenate([
            video_frames[[1], :, :, :],  # Blue
            video_frames[[0], :, :, :],  # Green
            video_frames[[2], :, :, :],  # Red
        ], axis=1)
    else:
        # bgr24
        video_frames = np.transpose(video_array.reshape(timesteps, width, height, 3), (3, 0, 1, 2))
    
    return video_frames, metadata, metaspec
