# -*- coding: utf-8 -*-
"""This module provides routines compress and decompress Environmental Video Array (.eva)-files"""


import os
import shutil


def dir_to_eva(input_dir, output_eva_file):
    """Compresses a dir into a zip renamed to a .eva file

    Args:
        input_dir (str): Path to a directory
        output_eva_file (str): output file name
    """
    shutil.make_archive(output_eva_file, 'zip', input_dir)
    os.rename(str(output_eva_file) + ".zip", str(output_eva_file))


def eva_to_dir(input_eva_file, output_dir):
    """Compresses a dir into a zip renamed to a .eva file

    Args:
        input_eva_file (str): Path to .eva file
        output_dir (str): output dir
    """
    shutil.unpack_archive(input_eva_file, output_dir, 'zip')
