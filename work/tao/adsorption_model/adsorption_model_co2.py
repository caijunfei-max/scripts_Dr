# -*-coding:utf-8 -*-
"""
# File       : adsorption_model.py
# Time       ：2024/3/19 9:07
# version    : 
# Author: Jun_fei Cai
"""

from pymatgen.core import Molecule
from pymatgen.analysis.adsorption import *
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.io.vasp.inputs import Poscar


struct = Structure.from_file("../structure/sqs_222.cif")
struct = SpacegroupAnalyzer(struct).get_conventional_standard_structure()
slabs = SlabGenerator(struct, miller_index=[1, 1, 1], min_slab_size=4.0,
                      min_vacuum_size=15.0, center_slab=False, lll_reduce=True)


adsorbate_1 = Molecule.from_file(filename="../adsorbate/Li2C2O4.mol")
adsorbate_2 = Molecule.from_file(filename="../adsorbate/Li2CO3.mol")

slab = slabs.get_slabs()[0]
# print(type(slab))
asf_h_entropy = AdsorbateSiteFinder(slab)
# ads_sites = asf_h_entropy.find_adsorption_sites()
ads_structures_li2c2o4 = asf_h_entropy.generate_adsorption_structures(adsorbate_1,
                                                                      repeat=None,
                                                                      min_lw= 10.0,
                                                                      find_args={"distance": 2.5,
                                                                                 "positions": "ontop"},
                                                                      translate=True)

ads_structures_li2co3 = asf_h_entropy.generate_adsorption_structures(adsorbate_2,
                                                                     repeat=None,
                                                                     min_lw= 10.0,
                                                                     find_args={"distance": 2.5,
                                                                                "positions": "ontop"},
                                                                     translate=True)

# ads_structures[0].to("POSCAR", fmt="poscar")
for i, j, k in zip(range(len(ads_structures_li2co3)), ads_structures_li2c2o4, ads_structures_li2co3):
    ads_li2c2o4 = Poscar(j)
    ads_li2co3 = Poscar(k)
    name_1 = "li2c2o4/"+"POSCAR_li2c2o4_" + str(i)
    name_2 = "li2co3/" + "POSCAR_li2co3_" + str(i)
    ads_li2c2o4.write_file(name_1)
    ads_li2co3.write_file(name_2)

