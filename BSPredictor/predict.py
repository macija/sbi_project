import os 
from openbabel import pybel
from PUResNet import PUResNet
import argparse
import numpy as np
import glob

def arg_parser():
    '''
    Add arguments to be used when running predict.py
    '''
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file_format','-ftype',required=True,type=str,help='File Format of Protein Structure like: mol2,pdb..etc. All file format supported by Open Babel is supported')
    parser.add_argument('--mode','-m',required=True,type=int,help='Mode 0 is for single protein structure. Mode 1 is for multiple protein structure')
    parser.add_argument('--input_path','-i',required=True,type=str,help='For mode 0 provide absolute or relative path for protein structure. For mode 1 provide absolute or relative path for folder containing protein structure')
    parser.add_argument('--output_format','-otype',required=False,type=str,default='mol2',help='Provide the output format for predicted binding side. All formats supported by Open Babel')
    parser.add_argument('--output_path','-o',required=False,type=str,default='output',help='path to model output')
    parser.add_argument('--gpu','-gpu',required=False,type=str,help='Provide GPU device if you want to use GPU like: 0 or 1 or 2 etc.')
    return parser.parse_args()



def get_coords(file_path, file_type):
    atom_coords = []
    residues = []
    with open(file_path, "r") as file:
        
        if file_type == "mol2":
            start = False
            for line in file:
                if "@<TRIPOS>ATOM" in line:
                    start = True
                elif "@<TRIPOS>BOND" in line:
                    start = False
                elif start == True:
                    x = float(line[17:26])
                    y = float(line[27:36])
                    z = float(line[37:46])
                    res = line[64:72].strip()
                    atom_coords.append([x,y,z])
                    residues.append(res)
                    
        elif file_type == "pdb":
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                        x = float(line[31:38])
                        y = float(line[39:46])
                        z = float(line[47:55])

                        res_num = line[23:27]
                        res_name = line[17:20]
                        res = (res_name + res_num).strip()
                        
                        atom_coords.append([x,y,z])
                        residues.append(res)
                
            
    return atom_coords, residues

def get_aminoacids(prot_path, out_path, prot_filetype, bs_filetype):  
    '''
    List all the aminoacids that are 4A in any direction from any of the
    points predicted as ligand by the program and save them in (?)
    '''
    #TODO - It will work differently for mol2 and for pdb. 
    


    for pocket in glob.glob(out_path+'/pocket*'):
        # Get coordinates for protein and predicted binding site    
        lig_coords, _ = get_coords(pocket, bs_filetype) 
        prot_coords, residues = get_coords(prot_path, prot_filetype) 
        
        # Get the coordinates of a box that comprises all ligand prediction points 
        # +-4A    
        lig_coords_m = np.matrix(lig_coords)
        x_min = lig_coords_m.min(0).item((0,0)) - 4
        y_min = lig_coords_m.min(0).item((0,1)) - 4
        z_min = lig_coords_m.min(0).item((0,2)) - 4
        x_max = lig_coords_m.max(0).item((0,0)) + 4
        y_max = lig_coords_m.max(0).item((0,1)) + 4
        z_max = lig_coords_m.max(0).item((0,2)) + 4
        
        # Check if protein coordinate is inside the box. If it is, check if euclidean
        # distance with any of ligand points is <4A
        
        nearby_residues = []
        for index,prot_atom in enumerate(prot_coords):
            x,y,z = prot_atom
            if x_min < x < x_max and y_min < y < y_max and z_min < z < z_max:
                for lig_atom in lig_coords:
                    dist = np.linalg.norm(np.asarray(prot_atom) - np.asarray(lig_atom))
                    if dist < 4:
                        nearby_residues.append(residues[index])
        nearby_residues = list(set(nearby_residues))
        print("nearby residues to the ligand binding site %s are:" %(os.path.basename(os.path.normpath(pocket))))
        print(nearby_residues)

        #### SAVE a .MOL2 WITH BINDING SITE
        orig_prot =next(pybel.readfile(prot_filetype,prot_path))
        non_bs_atoms = []
        for i in range(orig_prot.OBMol.NumAtoms()): #iterates over atoms of the original protein
            atom = orig_prot.OBMol.GetAtom(i+1)
            resname = atom.GetResidue().GetName()  #gets the residues of each atom
            if resname not in nearby_residues: #cheks if the residues are the ones we want 
                non_bs_atoms.append(atom)
        for atom in non_bs_atoms:
            orig_prot.OBMol.DeleteAtom(atom)
        output_bs =  pybel.Outputfile('mol2', out_path+'/bs_'+os.path.basename(os.path.normpath(pocket))) #put here the name of the file we want.
        output_bs.write(orig_prot)
        output_bs.close()



def main():

    # Check correct format of arguments
    args=arg_parser()
    if args.mode not in [0,1]:
        raise ValueError('Please Enter Valid value for mode')
    elif args.mode==0:
        if not os.path.isfile(args.input_path):
            raise FileNotFoundError('File Not Found')
    elif args.mode==1:
        if not os.path.isdir(args.input_path):
            raise FileNotFoundError('Folder Not Found')
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if args.file_format not in pybel.informats.keys():
        raise ValueError('Enter Valid File Format {}'.format(pybel.informats))
    if args.output_format not in pybel.outformats.keys():
        raise ValueError('Enter Valid Output Format {}'.format(pybel.outformats))
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
        
    # Predict binding sites
    model=PUResNet()
    model.load_weights('train_test_families_1rep_best_weights.h5')
    if args.mode==0:
        mol=next(pybel.readfile(args.file_format,args.input_path))
        o_path=os.path.join(args.output_path,os.path.basename(args.input_path))
        if not os.path.exists(o_path):
            os.mkdir(o_path)
        model.save_pocket_mol2(mol,o_path,args.output_format)
        # TODO - check how the save_pocket_mol2 handles the filenames

        get_aminoacids(args.input_path, o_path, args.file_format, args.output_format) #changed here the o_path
    
    elif args.mode==1:
        for name in os.listdir(args.input_path):
            mol_path=os.path.join(args.input_path,name)
            mol=next(pybel.readfile(args.file_format,mol_path))
            o_path=os.path.join(args.output_path,os.path.basename(args.mol_path))
            if not os.path.exists(o_path):
                os.mkdir(o_path)
            model.save_pocket_mol2(mol,o_path,args.output_format)
    

if __name__=='__main__':
    main()
