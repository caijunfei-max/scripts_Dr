# -*-coding:utf-8 -*-
"""
# File       : fulldoping.py
# Time       ：2023/8/29 10:15
# version    : 
# Author: Jun_fei Cai
# Description: 
"""
from itertools import combinations
from pymatgen.core import Structure
import random


ele_list = ["Ti", "V", "Cr", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd"]
path = "data/Li2MnO3_supercell.vasp"
struct = Structure.from_file(path)
binary_com = list(combinations(ele_list, 2))
ternary_com = list(combinations(ele_list, 3))
quaternary_com = list(combinations(ele_list, 4))
site_list = [i for i in range(24, 36)]


def replaces(structure, sites, element):
    # pymatgen的replace函数只能替换一个原子，本函数用于实现多个原子的替换
    for i in sites:
        structure.replace(i, element)
    return structure


print("begin unitary doping")
for i in ele_list:
    for j in site_list:
        struct_unitary = struct.replace(j, i)
        struct_name = "Li2{0}O3.vasp".format(i)
        struct_unitary.to(fmt="poscar", filename=struct_name)
print("unitary doping completed")

seed = 1
random.seed(seed)

print("begin binary doping")
# binary doping
site_binary = [i for i in range(1, 12)]
list_binary = []
for i in site_binary:
    for j in site_binary:
        if i + j == 12:
            list_binary.append(list([i, j]))
# def binary(structure, num):
#     # 输入一个结构以及a元素的数量
#     for i in binary_com:
#         element_a = i[0]
#         element_b = i[1]
#         a_site = random.sample(site_list, num)
#         b_site = list(set(site_list)-set(a_site))
#         if num == 6:
#             struct_binary = replaces(structure, a_site, element_a)
#             struct_binary = replaces(struct_binary, b_site, element_b)
#             name = "Li2{0}0.5{1}0.5.vasp".format(element_a, element_b)
#             struct_binary.to(fmt="poscar", filename=name)
#         else:
#             num1 = round(float(num/12), 2)
#             num2 = round(float(1 - num/12), 2)
#             struct_binary_1 = replaces(structure, a_site, element_a)
#             struct_binary_1 = replaces(struct_binary_1, b_site, element_b)
#             name_1 = "Li2{0}.vasp".format("".join([element_a, str(num1), element_b, str(num2)]))
#             struct_binary_1.to(fmt="poscar", filename=name_1)
#
#             struct_binary_2 = replaces(structure, a_site, element_b)
#             struct_binary_2 = replaces(struct_binary_2, b_site, element_a)
#             name_2 = "Li2{0}.vasp".format("".join([element_b, str(num1), element_a, str(num2)]))
#             struct_binary_2.to(fmt='poscar', filename=name_2)


# ternary
for i in list_binary:
    a_site = random.sample(site_list, i[0])
    b_site = random.sample(list(set(site_list)-set(a_site)), i[1])
    num1 = round(i[0]/12, 2)
    num2 = round(i[1]/12, 2)
    for com in binary_com:
        struct_binary = replaces(replaces(struct, a_site, com[0]), b_site, com[1])
        name_binary = "Li2{0}.vasp".format("".join([com[0], str(num1), com[1], str(num2)]))
        struct_binary.to(fmt='poscar', filename=name_binary)
    print("binary doping of {0} is completed".format(str(i)))


print("Begin ternary doping")
site_ternary = [i for i in range(1, 12)]
list_ternary = []
for i in site_ternary:
    for j in site_ternary:
        for k in site_ternary:
            if i + j + k == 12:
                list_ternary.append(list([i, j, k]))

for i in list_ternary:
    a_site = random.sample(site_list, i[0])
    b_site = random.sample(list(set(site_list)-set(a_site)), i[1])
    c_site = random.sample(list(set(site_list)-set(a_site)-set(b_site)), i[2])
    num1 = round(i[0] / 12, 2)
    num2 = round(i[1] / 12, 2)
    num3 = round(i[2] / 12, 2)
    for com in ternary_com:
        struct_ternary = replaces(replaces(replaces(struct, a_site, com[0]), b_site, com[1]), c_site, com[2])
        name_ternary = "Li2{0}.vasp".format("".join([com[0], str(num1), com[1], str(num2), com[2], str(num3)]))
        struct_ternary.to(fmt='poscar', filename=name_ternary)
    print("ternary doping of {0} is completed".format(str(i)))


print("Begin quatetnary doping")
site_quaternary = [i for i in range(1, 12)]
list_quaternary = []
for i in site_quaternary:
    for j in site_quaternary:
        for k in site_quaternary:
            for h in site_quaternary:
                if i + j + k + h == 12:
                    list_quaternary.append(list([i, j, k, h]))

for i in list_quaternary:
    a_site = random.sample(site_list, i[0])
    b_site = random.sample(list(set(site_list)-set(a_site)),
                           i[1])
    c_site = random.sample(list(set(site_list)-set(a_site)-set(b_site)),
                           i[2])
    d_site = random.sample(list(set(site_list)-set(a_site)-set(b_site)-set(c_site)),
                           i[3])
    num1 = round(i[0] / 12, 2)
    num2 = round(i[1] / 12, 2)
    num3 = round(i[2] / 12, 2)
    num4 = round(i[3] / 12, 2)
    for com in quaternary_com:
        struct_quaternary = replaces(replaces(replaces(replaces(
            struct, a_site, com[0]), b_site, com[1]), c_site, com[2]), d_site, com[3])
    print("quaternary doping of {0} is completed".format(str(i)))

print("all doping is competed")
