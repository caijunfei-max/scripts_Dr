# -*-coding:utf-8 -*-
"""
# File       : cifToVasp.py
# Time       ：2023/8/22 9:25
# version    : 
# Author: Jun_fei Cai
# Description: 
"""

from pymatgen.core import Structure
import os


all_files = os.listdir(".")
for file in all_files:
    if file.endwith(".cif"):
        structure = Structure.from_file(file)
        structure_name = file.split(".")[0]
        structure.to(fmt="poscar", filename="{0}.vasp".format(structure_name))