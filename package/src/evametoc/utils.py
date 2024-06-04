# -*- coding: utf-8 -*-
"""Module to provide some generic functions to this package."""


import glob
import os


def human_readable_file_size(num, suffix="B"):
    """Writes a number of bytes to a human-readable string

    Args:
        num (int): Number of bytes
        suffix (string): Unit to be placed after the suffix (default is "B" for bytes)
    Returns:
        human_readable (string): Number of bytes in a human-readable format (e.g. 154.7GiB)
    """
    if num is None:
        return "Unconstrained"
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def find_target_path(source_path, suffix, target_path=None, overwrite=False):
    """Generates a path where to store a file, based on a source-file path.

    Args:
        source_path (str): path of the source file
        suffix (str): extension (with suffix) of the target file
        target_path (str or None): optional path that a user may have supplied.
            If not None, uses this path instead of source_path and suffix
        overwrite (bool): Overwrite an existing file
            If False, appends the target_path with a number

    Returns:
        target_path (str): a location where the target file may be written.
    """
    if target_path is not None:
        return target_path
    
    source_path_root, source_path_ext = os.path.splitext(source_path)
    target_path = source_path_root + suffix
    
    if not os.path.exists(target_path):
        return target_path
    if overwrite:
        return target_path
    
    suffix_start, suffix_ext = suffix.rsplit('.', 1)
    existing_files = glob.glob(source_path_root + suffix_start + '.*.' + suffix_ext)
    existing_files_numbers = list(map(lambda fn: int(fn.rsplit('.', 3)[-2]), existing_files))
    if existing_files_numbers:
        new_number = max(existing_files_numbers) + 1
        return source_path_root + suffix_start + f'.{new_number:d}.' + suffix_ext
    else:
        return source_path_root + suffix_start + '.1.' + suffix_ext


def sort_dict_by_keys(dictionary):
    """Sorts a dictionary by its keys

    Args:
        dictionary (dict): input dictionary
    Returns:
        sorted (dict): a copy of dictionary sorted by its keys
    """
    return dict(sorted(dictionary.items(), key=lambda d: d[0]))
