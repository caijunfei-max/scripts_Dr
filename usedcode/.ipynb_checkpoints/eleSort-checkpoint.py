# -*- coding: utf-8 -*-
"""
created on 2023/9/3, %+23:51
Author: Junfei Cai
# 对结构进行各种替换之后，总是会出现POSCAR中原子乱排，这个文件是为了对POSCAR重新排列一下，课题一只对过渡金属进行排列
"""

from pymatgen.core import Structure


poscar_dict = Structure.from_file("data/POSCAR").as_dict()
sites_list = poscar_dict["sites"]

sorted_sites = sorted(sites_list, key=lambda x: x["label"])
poscar_dict["sites"] = sorted_sites
poscar = Structure.from_dict(poscar_dict)

poscar.to(fmt="poscar", filename="POSCAR")
