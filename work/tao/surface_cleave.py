# -*-coding:utf-8 -*-
"""
# File       : surface_cleave.py
# Time       ：2024/3/18 10:53
# version    : 
# Author: Jun_fei Cai
# Description: 
"""
from pymatgen.analysis.adsorption import *
from pymatgen.core.surface import Slab, SlabGenerator, generate_all_slabs, Structure, Lattice, ReconstructionGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from matplotlib import pyplot as plt
from pymatgen.io.vasp.inputs import Poscar


struct = Structure.from_file("structure/pt.vasp")
struct = SpacegroupAnalyzer(struct).get_conventional_standard_structure()
slab = SlabGenerator(struct, miller_index=[1, 1, 1], min_slab_size=8.0,
                     min_vacuum_size=15.0, center_slab=True)

for n, slabs in enumerate(slab.get_slabs()):
    slabs_bak = slabs.copy()
    slabs.make_supercell([[2, 0, 0],
                          [0, 2, 0],
                          [0, 0, 1]])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_slab(slabs, ax, adsorption_sites=True)
    plt.savefig(str(n) + '-Au-111.png', format='png')
    open('POSCAR' + '-' + str(n), 'w').write(str(Poscar(slabs)))
