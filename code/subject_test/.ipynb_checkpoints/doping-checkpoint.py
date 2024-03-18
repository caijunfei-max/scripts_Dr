# -*-coding:utf-8 -*-
"""
# File       : li2mno3_doping.py
# Time       ：2023/8/24 18:44
# version    :
# Author: Jun_fei Cai
# Description:
"""
from pymatgen.core import Structure
from itertools import combinations, permutations


def unitary_dop(struct, ele_list, site):
    # struct should be a Structure object，
    # ele_list should be a list includes all atoms for doping
    # site is str or list includs sites to be replaced.
    # This method is used to perform doping of one element
    if type(site) == int:
        for ele in ele_list:
            struct.replace(site, ele)
            struct_name = "Li2MnO3_{0}.vasp".format(ele)
            struct.to(fmt="poscar", filename=struct_name)
    elif type(site) == list:
        for ele in ele_list:
            for i in site:
                struct.replace(i, ele)
                struct_name = "Li2MnO3_{0}_{1}.vasp".format(ele, str(len(site)))
                struct.to(fmt="poscar", filename=struct_name)


def binary_dop(struct, ele_list, site1, site2):
    # performing binary_dop on a compound
    # struct:pymatgen.core.Structure site1,site2:list ele_list:list
    if len(site1) == len(site2):
        binary_com = combinations(ele_list, 2)
        for com in binary_com:
            for i in site1:
                struct.replace(i, com[0])
            for j in site2:
                struct.replace(j, com[1])
            struct_name = "Li2MnO3_{0}_{1}_{2}_{3}.vasp".format(com[0], len(site1), com[1], len(site2))
            struct.to(fmt="poscar", filename=struct_name)
    else:
        binary_com = permutations(ele_list, 2)
        for com in binary_com:
            for i in site1:
                struct.replace(i, com[0])
            for j in site2:
                struct.replace(j, com[1])
            struct_name = "Li2MnO3_{0}_{1}_{2}_{3}.vasp".format(com[0], len(site1), com[1], len(site2))
            struct.to(fmt="poscar", filename=struct_name)


def ternary_dop(struct, ele_list, site1, site2, site3):
    # performing ternary_dop on a compound
    # struct:pymatgen.core Structure, ele_list:list, site1,site2,site3:int
    ternary_com = combinations(ele_list, 3)
    for com in ternary_com:
        struct.replace(site1, com[0])
        struct.replace(site2, com[1])
        struct.replace(site3, com[2])
        struct_name = "Li2MnO3_{0}_{1}_{2}".format(com[0], com[1], com[2])
        struct.to(fmt="poscar", filename=struct_name)
