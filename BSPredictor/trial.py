# Imports of data
from math import ceil
import numpy as np
from data import Featurizer, make_grid
from openbabel import pybel

# Import files

# Start featurizer and import molecule
featurizer=Featurizer(save_molecule_codes=False)

# pybel can read mol2 and pdb files. Users should have the option to check both
mol=next(pybel.readfile("mol2","protein.mol2"))
bs=next(pybel.readfile("mol2","ligand.mol2"))

# Hyperparameters (?)
scale=0.5
max_dist=35
# def pocket_density_from_mol(self, mol):
prot_coords, prot_features = featurizer.get_features(mol)
#       prot_coords:        nº prots x 3D (XYZ)
#       prot_features:      nº prots x 18D (the 18 features, in a 0,1 array)

lig_coords, _ = featurizer.get_features(bs)

#lig_features is an array of 1s (probability of 1 of there being a bs)
# It has to be a 2D array of N x 1F (1 feature: the probability)
lig_features = np.ones((len(lig_coords), 1))

# Get the center of the protein
centroid = prot_coords.mean(axis=0)
# Make all coordinates be respect the center of the protein
prot_coords -= centroid

resolution = 1. / scale

# make_grid is also from data. Could do it myself.
mol_grid = make_grid(prot_coords, prot_features,
                            max_dist=max_dist,
                            grid_resolution=resolution)

bs_grid = make_grid(lig_coords, lig_features,
                            max_dist=max_dist,
                            grid_resolution=resolution)


#       x:          (1, 36, 36, 36, 18)

print(bs_grid)
print(mol_grid.shape)

# This x is what is fed into the program:
#density = self.predict(x)
origin = (centroid - max_dist)
step = np.array([1.0 / scale] * 3)
