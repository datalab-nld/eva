# -*- coding: utf-8 -*-
"""This module provides routines to read and write the metadata alongside the video files"""

import datetime
import dateutil
import json
import re
import os
import numpy as np


def write(video_file, metadata, separate_metadata=False):
    """Writes/appends the metadata for one video file to a json-file.

    Writes the metadata for one video file to a json file, in the same folder as the video file.
    This JSON-file contains the scaling data for the video file, and other data needed
    for recreation of the original data. If a meta.json already exists, the data will be appended.

    Args:
        video_file (str): Path to the video file, in which folder the metadata will be placed.
        metadata (dict): Meta-data for this video file.
        separate_metadata (bool): if true, uses separate metadata files for each video,
            if false, writes all metadata to one meta.json file.
    """
    assert all(k in metadata.keys() for k in "BGR"), "Metadata must have keys B, G or R"

    # Check for any special/root metadata-keys, like _shared_coords_
    metadata_special = {}
    metadata_video = {}
    for key in list(metadata.keys()):
        if key.startswith('_') and key.endswith('_'):
            metadata_special[key] = metadata[key]
        else:
            metadata_video[key] = metadata[key]

    # Get path of the metadata-file
    metafile = os.path.join(os.path.dirname(video_file), 'meta.json')
    if separate_metadata:
        metafile = video_file + '.json'

    # Generate the dict to be added to the metadata-file
    meta_new = {os.path.basename(video_file): metadata_video}
    meta_new.update(metadata_special)

    # Write the metadata to file, or join with existing metadata
    if not os.path.exists(metafile):
        with open(metafile, 'w') as fh:
            json.dump(meta_new, fh, cls=NumpyJsonEncoder, indent=4)
    else:
        with open(metafile, 'r+') as fh:
            meta = json.load(fh, cls=NumpyJsonDecoder)
            fh.seek(0)

            # Make sure new video-files are added, and specials (like coords) are updated
            for key in set(list(meta.keys()) + list(meta_new.keys())):
                if key not in meta:
                    meta[key] = meta_new[key]
                elif key in meta_new:
                    if key.startswith('_') and key.endswith('_'):
                        meta[key].update(meta_new[key])
                    else:
                        meta[key] = meta_new[key]
            json.dump(meta, fh, cls=NumpyJsonEncoder, indent=4)


def find(video_file):
    """Searches for the metadata file in the dir of a video file

    Args:
        video_file (str): path of the video file, for which the metadata must be found.
    Returns:
        metafile (str): path of the metadata-file
    Raises:
        OSError: if the file could not be found
    """
    if os.path.isfile(video_file + '.json'):
        metafile = video_file + '.json'
        return metafile
    else:
        metafile = os.path.join(os.path.dirname(video_file), 'meta.json')
        if os.path.isfile(metafile):
            with open(metafile, 'r') as fh:
                metadata = json.load(fh, cls=NumpyJsonDecoder)
            if os.path.basename(video_file) in metadata:
                return metafile
            raise OSError(61, "No data available", metafile)
        else:
            raise OSError(2, "No such file or directory", metafile)


def read(video_file, separate_metadata=None):
    """Reads the metadata for a certain video file

    Args:
        video_file (str): path to the video file, for which the metadata must be read.
        separate_metadata (bool): if true, assumes separate metadata files for each video,
            if false, assumes all metadata in one meta.json file,
            if None, searches for the metadata-file using find_metadata
    Returns:
        metadata (dict): metadata for this video file
        metaspec (dict): global metadata (such as _shared_coords_) present in the metadata-file
    """
    # Define metadata-filename
    if separate_metadata is None:
        metafile = find(video_file)
    elif separate_metadata:
        metafile = video_file + '.json'
    else:
        metafile = os.path.join(os.path.dirname(video_file), 'meta.json')

    # Open the metadata
    with open(metafile, 'r') as fh:
        metadata = json.load(fh, cls=NumpyJsonDecoder)

    # Extract the video file-specific and global metadata
    metaspec = {key: v for key, v in metadata.items() if key.startswith('_') and key.endswith('_')}
    return metadata[os.path.basename(video_file)], metaspec


class NumpyJsonEncoder(json.JSONEncoder):
    """Extension to the default JSON encoder that also encode numpy.arrays, numpy.floats and numpy.datetime64[ns]

    | Numpy          | Python            | JSON   |
    |----------------|-------------------|--------|
    | ndarray        | list              | array  |
    | float32        | float             | number |
    | float64        | float             | number |
    | datetime64[ns] | string            | string |
    | -              | datetime.datetime | string |

    Datetimes are encoded as %Y-%m-%dT%H:%M:%S%z.
    Other data types (not defined above) are encoded using the regular json.JSONEncoder

    The numpy module does not provide this functionality, so it is included here
    """

    def default(self, obj):
        """This method returns a serializable object for ``obj``"""
        if isinstance(obj, np.ndarray):
            if 'datetime64[ns]' in obj.dtype.name:
                # Datetimes are encoded as int in ns, but that notation is unintuitive and
                # that precision is rarely needed
                return obj.astype('datetime64[s]').astype('str').tolist()
            else:
                return obj.tolist()
        if np.issubdtype(obj, np.floating):
            return float(obj)
        if np.issubdtype(obj, np.integer):
            return int(obj)
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%dT%H:%M:%S%z")
        return json.JSONEncoder.default(self, obj)


class NumpyJsonDecoder(json.JSONDecoder):
    """Extension of the default json.JSONDecoder that convert strings back to datetime.datetime-s"""

    datetime_regex_str = r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}'

    def __init__(self, *args, **kwargs):
        super(NumpyJsonDecoder, self).__init__(*args, **kwargs)
        self.parse_string = self.parse_string_or_date
        self.scan_once = json.decoder.scanner.py_make_scanner(self)
        self.dtregex = re.compile(self.datetime_regex_str)

    def parse_string_or_date(self, s, *args, **kwargs):
        """Replacement for the default parse_string method. Looks for strings that matches the self.dtregex"""
        s, end = json.decoder.scanstring(s, *args, **kwargs)  # Call to the original method
        if self.dtregex.match(s):
            # If it looks like a date, it is probably a date. Try to decode
            try:
                s = dateutil.parser.parse(s)
            except (dateutil.parser.ParserError, dateutil.parser.UnknownTimezoneWarning):
                pass
        return s, end
