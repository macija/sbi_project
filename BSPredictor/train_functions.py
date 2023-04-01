import os
import numpy as np
import keras
import keras.backend as K
from openbabel import pybel, openbabel
from sklearn.model_selection import train_test_split

from data import Featurizer, make_grid
from MyModel import PUResNet


def get_grids(file_type, prot_input_file, bs_input_file=None,
              grid_resolution = 2, max_dist = 35, 
              featurizer=Featurizer(save_molecule_codes=False)):
    '''
    Converts both a protein file (PDB or mol2) and its ligand (if specified)
    to a grid.
        
    Parameters
    ----------
    file_type: "pdb", "mol2"
    prot_input_file, ligand_input_file: protein and ligand files
    grid_resolution: float, optional
        Resolution of a grid (in Angstroms).
    max_dist: float, optional
        Maximum distance between atom and box center. Resulting box has size of
        2*`max_dist`+1 Angstroms and atoms that are too far away are not
        included.
    '''
    
    # Convert file into pybel object and get the features of the molecule. 
    # If binding site, features is an array of 1s (indicating that bs is present)
    
    # TODO - pybel gives a lot of warnings. I should check them and maybe remove some molecules
    # or see how much does it affect the model
    prot = next(pybel.readfile(file_type,prot_input_file))
    prot_coords, prot_features = featurizer.get_features(prot)
    
    # Change all coordinates to be respect the center of the protein
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid
    # Create the grid
    prot_grid = make_grid(prot_coords, prot_features,
                        max_dist=max_dist,
                        grid_resolution=grid_resolution)
    
    
    # Do the same for the binding site, if input file specified
    if bs_input_file != None:
        bs = next(pybel.readfile(file_type, bs_input_file))
        bs_coords, _ = featurizer.get_features(bs)
        # BS just has 1 feature: an array of 1s for each atom, indicating the
        # atom is present in that position
        bs_features = np.ones((len(bs_coords), 1))
        bs_coords -= centroid
        bs_grid = make_grid(bs_coords, bs_features,
                    max_dist=max_dist,
                    grid_resolution=grid_resolution)
        print()
    else:
        bs_grid = None
    
    return prot_grid, bs_grid, centroid

def get_training_data(input_folder):
    """
    Returns a np array containing the protein grids, one np array with the binding_sites grids,
    and the ceentroid coordinates for each one. 
    """
    proteins = None
    binding_sites = None
    centroids = []
    for root, dirs, _ in os.walk(input_folder, topdown=False):
        for dir in dirs:

            protein_file = os.path.join(root, dir, "protein.mol2")
            ligand_file = os.path.join(root, dir, "ligand.mol2")
            
            prot_grid, bs_grid, centroid = get_grids("mol2", protein_file, ligand_file)
            
            if proteins is None:
                proteins = prot_grid
                binding_sites = bs_grid
            else:
                proteins = np.concatenate((proteins, prot_grid), axis=0)
                binding_sites = np.concatenate((binding_sites, bs_grid), axis=0)
            
            # Append to proteins and binding sites list the grids of proteins and bs
            centroids.append(centroid)
        
    
    print("Number of proteins to train the model:", proteins.shape[0])
    # TODO - Centroids are completely unnecessary to maintain
    return proteins, binding_sites, centroids

proteins, binding_sites, _ = get_training_data("scPDB_similarproteins")



def DiceLoss(targets, inputs, smooth=1e-6):
    '''
    Loss function to use to train the data
    call with: model.compile(loss=Diceloss)
    DiceLoss is used as it was
    '''
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #reshape to 2D matrices
    inputs = K.reshape(inputs, (-1, 1))
    targets = K.reshape(targets, (-1, 1))
    
    intersection = K.sum(targets * inputs)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


X_train, X_test, y_train, y_test = train_test_split(proteins, 
                                                    binding_sites, 
                                                    test_size=0.33, 
                                                    random_state=42)