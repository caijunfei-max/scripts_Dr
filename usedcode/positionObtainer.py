# -*-coding:utf-8 -*-
"""
# File       : wyckoffPositionObtainer.py
# Time       ：2023/8/21 9:35
# version    : 
# Author: Jun_fei Cai
# Description: Functions defined here is used to obtain the Wyckoff-position of a structure.
# 只要在脚本末尾position_df的变量定义部分输入相应的path就行，必须是POSCAR类型的文件
# 输出是一个dataframe，前四列为原子类型和坐标，最后一列是位点名称。
"""

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd
import numpy as np


def site_info(structure):
    # 返回原子信息和原子坐标
    site_list = []
    coordinates_list = []
    for i in range(structure.num_sites):
        site_list.append(structure[i].species)
        coordinates_list.append(structure[i].frac_coords)

    site_array = np.array(site_list)
    coords_array = np.array(coordinates_list)

    return site_array, coords_array


def duplicate_num(list_mul):
    # 返回一个列表中元素的重复次数
    # list_mul应该为列表
    duplicate_num_list = []
    for element in list_mul:
        duplicate_times = len([a for a in list_mul if a == element])
        duplicate_num_list.append(duplicate_times)
    return duplicate_num_list


def wyckoff_position(structure):
    # return the wyckoff position of each site of structure, which is a pymatgen.core.Structure
    # 返回一个数组里面是wyckoff符号
    symmetry_ana = SpacegroupAnalyzer(structure)

    equiv_site = symmetry_ana.get_symmetry_dataset()["equivalent_atoms"]
    multiplicity_list = duplicate_num(equiv_site)

    wyckoff_letter = symmetry_ana.get_symmetry_dataset()["wyckoffs"]

    position_list = []
    for i in range(structure.num_sites):
        position = "".join([str(multiplicity_list[i]), wyckoff_letter[i]])
        position_list.append(position)
    position_array = np.array(position_list)

    # 对数组进行升维,因为前面原子类型和原子坐标是二维数组，为了合成dataframe这里一样升到二维数组
    position_array_2d = position_array[:, np.newaxis]

    return position_array_2d


def position_dfer(path):
    # 将wickoff位置符号封装到dataframe中
    structure = Structure.from_file(path)
    element, xyz = site_info(structure)
    position = wyckoff_position(structure)
    position_array = np.hstack((element, xyz, position))
    wyckoffposition_info = pd.DataFrame(position_array, columns=["element", "x", "y", "z", "Wyckoff position"])

    return wyckoffposition_info


position_df = position_dfer("path")


