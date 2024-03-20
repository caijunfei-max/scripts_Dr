# -*-coding:utf-8 -*-
"""
# File       : adsorption_model.py
# Time       ：2024/3/19 9:07
# version    : 
# Author: Jun_fei Cai
"""
# %%
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.adsorption import *
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.io.vasp.inputs import Poscar

# %%
struct = Structure.from_file("structure/sqs_222.cif")
struct = SpacegroupAnalyzer(struct).get_conventional_standard_structure()
slabs = SlabGenerator(struct, miller_index=[1, 1, 1], min_slab_size=6.0,
                      min_vacuum_size=15.0, center_slab=False, lll_reduce=True)

# %%
# top site model of
adsorbate_co2 = Molecule.from_file(filename="adsorbate/CO2.mol")
slab = slabs.get_slabs()[0]
# print(type(slab))
asf_h_entropy = AdsorbateSiteFinder(slab)
ads_sites = asf_h_entropy.find_adsorption_sites(positions=["ontop"])
ads_structures_top = asf_h_entropy.generate_adsorption_structures(adsorbate_co2,
                                                              repeat=[2, 2, 1],
                                                              find_args={"positions":["ontop"]},
                                                              translate=False)

# ads_structures[0].to("POSCAR", fmt="poscar")
for i, j in zip(range(len(ads_structures_top)), ads_structures_top):
    ads_poscar = Poscar(j)
    name = "POSCAR" + str(i)
    ads_poscar.write_file(name)

