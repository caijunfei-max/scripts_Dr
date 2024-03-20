from pymatgen.core import Molecule

a = Molecule.from_file("adsorbate/CO2.mol")
print(type(a))
