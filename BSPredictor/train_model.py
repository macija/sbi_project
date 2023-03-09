# Imports
#from MyModel import MyModel
import os
import numpy as np
from data import Featurizer, make_grid
from openbabel import pybel, openbabel


# Funct 1: Get the protein ( + ligand, if specified ) grid
# Funct 2: Knowing the center of the protein (stored in an array), recreate the ligand

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


def bs_grid_to_mol2(centroid, bs_grid, output_folder, file_type, 
                    grid_resolution = 2, max_dist = 35):
    
    '''
    Converts the binding site grid into a molecule file to be opened with a visual program.
    Saves the molecule file in the specified folder
    
    Parameters
    ----------
    centroid: the center of the protein [x,y,z]
    bs_grid: grid to convert to molecule file
    output_folder: folder where the file will be stored as BindingSite.<file_type>
    file_type: "pdb", "mol2"
    grid_resolution, max_dist: the ones specified in get_grids function
    '''
    
    # Get all the coordinates in the matrix that do not have a 0
    # It may get less coords than original atoms: some atoms get collapsed into the same
    # grid point
    grid_coords = np.argwhere(bs_grid)[:,1:4]

    # Shift the coordinates to align with the coordinates of the input protein
    coords = grid_coords.astype('float32') * grid_resolution - max_dist + centroid
    
    # Create the binding site file and save it to the specified folder
    bs = openbabel.OBMol()
    for x,y,z in coords:
        a = bs.NewAtom()
        a.SetVector(float(x),float(y),float(z))
    p_bs = pybel.Molecule(bs)
    
    # Create folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    p_bs.write(file_type,output_folder+'/BindingSite.'+file_type, overwrite=True)


    # TODO - get density for each coord (nºatoms per non-empty coord). (idk if its necessary for anything tbh)
    density = [bs_grid[0, coord[0], coord[1], coord[2]] for coord in grid_coords]



def get_training_data(input_folder):
    proteins = []
    binding_sites = []
    centroids = []
    for root, dirs, _ in os.walk(input_folder, topdown=False):
        for dir in dirs:
            print(dir)
            protein_file = os.path.join(root, dir, "protein.mol2")
            ligand_file = os.path.join(root, dir, "ligand.mol2")
            
            prot_grid, bs_grid, centroid = get_grids("mol2", protein_file, ligand_file)
            
            # Append to proteins and binding sites list the grids of proteins and bs
            proteins.append(prot_grid)
            binding_sites.append(bs_grid)
            centroids.append(centroid)
    
    #Centroids are completely unnecessary to maintain
    return proteins, binding_sites, centroids
    


def main():
    '''
    Run through the different folders
    '''

    # Get the data
    proteins, binding_sites, centroids = get_training_data("scPDB_100_random")
    
    bs_grid_to_mol2(centroids[0], binding_sites[0], "output", "mol2")
    
    # Train the model. Proteins is X, binding_sites is Y      
    #MyModel.train(proteins, binding_sites)


if __name__=='__main__':
    main()


            



#print(indices)
#print(len(indices))