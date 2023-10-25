"""
# File       : plusU.py
# Time       ：2023/8/27 11:41
# Author: Jun_fei Cai
# Description: Automatically GGA+U editor
# 路径内应该有INCAR和POSCAR，此脚本运行后可以根据POSCAR的情况对INCAR进行加U。
"""
# from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure

uValue = {"Ni": 6.0, "Mn": 3.9,  "Ru": 4.0, "Mo": 4.4, "Nb": 1.5, "Co": 3.32, "Fe": 5.3, "Cr": 3.7}
path1 = "./POSCAR"
path2 = "./INCAR"


def d_switch(element):
    # 返回是否开U
    if "d" in element.electronic_structure:
        return True
    else:
        return False


def u_value(element):
    # 设置默认值
    if d_switch(element):
        if element.symbol in uValue.keys():
            return uValue[element.symbol]
        else:
            return 3.0
    else:
        return 0


def ele_from_struct(structure):
    species_list = []
    # 从结构中生成元素，输入数据应该是一个pymatgen.core.Structure类
    for i in structure.species:
        if i in species_list:
            continue
        else:
            species_list.append(i)
    return species_list


def pluslist(structure):
    # 返回加u设置的字符串放在列表中
    uswitch = "LDAU = True"

    utype = "LDAUTYPE = 2"

    on_off = []
    for i in ele_from_struct(structure):
        if d_switch(i):
            on_off.append("2")
        else:
            on_off.append("-1")
    u_on_off = "LDAUL = {0}".format(" ".join(on_off))

    value_u = []
    value_j = []
    for ele in ele_from_struct(structure):
        value_u.append(str(u_value(ele)))
        value_j.append("0")
    u = "LDAUU = {0}".format(" ".join(value_u))
    j = "LDAUJ = {0}".format(" ".join(value_j))

    ustr_list = [uswitch, utype, u_on_off, u, j]
    return ustr_list


poscar = Structure.from_file(path1)
with open(path2, "a") as f:
    for parameter in pluslist(poscar):
        f.write(parameter+"\n")
    f.close()
