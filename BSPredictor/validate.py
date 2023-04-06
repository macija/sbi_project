import os 
from PUResNet import PUResNet
import numpy as np
from openbabel import pybel
from data import Featurizer, make_grid
from predict import get_coords


def get_bs_coords(file_type, actual_bs_file, pred_bs_file, 
                featurizer=Featurizer(save_molecule_codes=False)):
    
    """
    Get coordinates for actual and predicted binding site as numpy arrays
    """
    
    # Get actual bs grid centered on 0
    actual_bs =  next(pybel.readfile(file_type, actual_bs_file))
    actual_bs_coords, _ = featurizer.get_features(actual_bs)
    
    # Get predicted bs grid centered on actual grid
    pred_bs_coords, _ = get_coords(pred_bs_file, file_type)
    pred_bs_coords = np.array(pred_bs_coords)
    
    return actual_bs_coords, pred_bs_coords


           
def get_predicted_bs(prot_format, prot_path, o_path, weights_path):
    """
    Return the predicted binding site for a given protein
    """
    model=PUResNet()
    model.load_weights(weights_path)
    
    prot = next(pybel.readfile(prot_format, prot_path))
    
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    model.save_pocket_mol2(prot,o_path,"mol2")
    # I have to somehow check all pockets, not only pocket0
    
    predicted_bs_filepath = os.path.join(o_path,"pocket0.mol2")
    return predicted_bs_filepath    
 


def calculate_DVO(actual_bs_coords, pred_bs_coords):
    """
    Calculate the Dice Volume Overlap (DVO) between two groups of coordinates:
    DVO = intersection / union

    To calculate the DVO, the coordinates are first rounded to the nearest integer values and then a grid 
    of points is generated covering the bounding boxes of the two groups of coordinates. All the interior 
    points are filled with new coordinates, so that the entire inside of the grid is full of points.
    
    Then, the intersection and union of the two sets of points are calculated, from where DVO is derived.

    The resulting DVO value provides a measure of the similarity between the two bounding boxes, with a 
    value of 1 indicating perfect overlap and a value of 0 indicating no overlap.
    """
    
    # Create grid of integers
    actual_bs_coords = actual_bs_coords.round().astype(int) 
    pred_bs_coords = pred_bs_coords.round().astype(int) 
    
 
    # Fill in the grid: ensure that all interior points of the grid are filled
    for i, bs_coords in enumerate([actual_bs_coords, pred_bs_coords]):
        
        # Generate a grid of points covering the bounding box of bs_coords
        xmin, ymin, zmin = bs_coords.min(axis=0)
        xmax, ymax, zmax = bs_coords.max(axis=0)
        xgrid, ygrid, zgrid = np.meshgrid(range(xmin, xmax+1), range(ymin, ymax+1), range(zmin, zmax+1), indexing='ij')
        grid_coords = np.stack((xgrid.ravel(), ygrid.ravel(), zgrid.ravel()), axis=1)

        # Remove the points that are already in bs_coords
        new_coords = grid_coords[~np.in1d(grid_coords.view(dtype=[('', bs_coords.dtype)]*3), bs_coords.view(dtype=[('', bs_coords.dtype)]*3))]

        # Concatenate the new coordinates with bs_coords
        bs_coords = np.concatenate((bs_coords, new_coords))
        
        # Update actual_bs_coords and pred_bs_coords
        if i == 0:
            actual_bs_coords = bs_coords
        if i == 1:
            pred_bs_coords = bs_coords            
 
    # Convert the coordinates to sets of tuples for efficient intersection and union calculations
    actual_bs_coords_set = set([tuple(x) for x in actual_bs_coords.tolist()])
    pred_bs_coords_set = set([tuple(x) for x in pred_bs_coords.tolist()])
    
    # Calculate the intersection and union of the two sets of coordinates
    intersect = len(actual_bs_coords_set.intersection(pred_bs_coords_set))
    union = len(actual_bs_coords_set.union(pred_bs_coords_set))
    
    # Calculate the Dice Volume Overlap (DVO)
    dvo = intersect / union
    return dvo
            
    
def validate(val_subset, output_folder, weights_path):
    total_val_proteins = 0
    predicted_pockets = 0
    dcc_success = 0
    dvo_vals = []
    for root, dirs, _ in os.walk(val_subset, topdown=False):
        for dir in dirs:
            total_val_proteins +=1
            
            # Get the file paths for the cavity and the protein to predict the binding site
            protein_file = os.path.join(root, dir, "protein.mol2")
            actual_bs_file = os.path.join(root, dir, "cavity6.mol2")
            
            # If cavity has been predicted
            predicted_bs_file = get_predicted_bs("mol2", protein_file, os.path.join(output_folder, dir), weights_path)
            
            # Check if any cavity has been predicted
            if os.path.exists(predicted_bs_file):
                predicted_pockets += 1
            else:
                continue 
            
            # Get the grids
            actual_bs_coords, pred_bs_coords = get_bs_coords("mol2", actual_bs_file, predicted_bs_file)
            
            # Calculate DCC:  
            dcc = np.linalg.norm(actual_bs_coords.mean(axis=0) - pred_bs_coords.mean(axis=0))
            
            if dcc < 4:
                dcc_success += 1
                dvo = calculate_DVO(actual_bs_coords, pred_bs_coords)
                dvo_vals.append(dvo)
    
    print("··············SUMMARY··············")
    print(f"Predicted pockets:       {predicted_pockets / total_val_proteins}") 
    print(f"Predictions with DCC<4A: {dcc_success / total_val_proteins}")           
    print(f"Mean value for DVO:      {sum(dvo_vals) / len(dvo_vals)}")    
    

if __name__=='__main__':
    validate("../../TrainData/val_subset", "output", "../../TrainData/train_test_families_1rep_best_weights.h5")