import openbabel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
import seaborn as sns
import openbabel.pybel as pybel
import glob
import itertools
from IPython.display import SVG
import sys

def calculate_fingerprints(protein_path, file_type):
    fingerprint_l = []
    mol = next(pybel.readfile(file_type, protein_path))
    residues = [x for x in openbabel.OBResidueIter(mol.OBMol)]
    for i in range(len(residues)-3):
        window = residues[i:i+3]
        window_res = openbabel.OBMol()
        for r in window:
            for atom in openbabel.OBResidueAtomIter(r):
                window_res.AddAtom(atom)
        window_res.ConnectTheDots()
        mol = pybel.Molecule(window_res)
        maccs_key = list(mol.calcfp('MACCS').bits)
        fingerprint = [1 if bit in maccs_key else 0 for bit in range(167)]
        fingerprint_l.append(fingerprint)
        return fingerprint_l

def calculate_tanimoto_coef(fingerprint1, fingerprint2):
    fp1_a = np.asarray(fingerprint1)
    fp2_a = np.asarray(fingerprint2)
    intersect = np.sum(fp1_a*fp2_a)
    tanimoto = intersect / (np.sum(fp1_a)+np.sum(fp2_a)-intersect)
    return tanimoto

def get_tanimoto(filetype1, file1, filetype2, file2):
    finger1 = calculate_fingerprints(file1, filetype1)
    finger2 = calculate_fingerprints(file2, filetype2)
    if len(finger1) == len(finger2):
        tanimoto = calculate_tanimoto_coef(finger1, finger2)
        return tanimoto
    else:
        #this shouldn't be happening
        print("MACCS have different length in %s, %s" %(file1, file2))
        l_1 = len(finger1)
        l_2 = len(finger2)
        if l_1 > l_2:
            tanimotos_l = []
            for i in range(l_1-l_2):
                tani = calculate_tanimoto_coef(finger1[i:i+l_2], finger2)
                tanimotos_l.append(tani)
            return np.max(tanimotos_l)
        else:
            tanimotos_l = []
            for i in range(l_2-l_1):
                tani = calculate_tanimoto_coef(finger1, finger2[i:i+l_1])
                tanimotos_l.append(tani)
            return np.max(tanimotos_l)

def calculate_cluster(list_pdbs):
    tanimotos_l = []
    for i,j in itertools.combinations(list_pdbs, 2):
        temp_tan = []
        for folder1 in glob.glob('../db/scPDB/'+i+'*/'):
            for folder2 in glob.glob('../db/scPDB/'+j+'*/'):
                tanimoto = get_tanimoto('mol2', folder1+'protein.mol2', 
                                        'mol2', folder2+'protein.mol2')
                temp_tan.append(tanimoto)
        tanimotos_l.append((i, j, np.max(temp_tan)))
    return tanimotos_l

if __name__=="__main__":
    if len(sys.argv) != 2:
        raise BaseException('Arguments incorrect')
    else:
        csv_to_read = sys.argv[1]
        clust_df = pd.read_csv(csv_to_read, sep=';')   
        clust_df['PDB_codes'] = clust_df['PDB_codes'].apply(lambda x: ast.literal_eval(x))
    
    in_file = sys.argv[1].split('/')[1]
    in_file = in_file.split('.')[0]
    os.mkdir('clusts_out/'+in_file)
    
    cluster_to_evaluate = []
    for cluster in clust_df.iterrows():
        if cluster[1][2] == 1:
            continue
        cluster_name = cluster[1][0]
        cluster_l = cluster[1][1]
        tanimotos_l = calculate_cluster(cluster_l)
        tanimotos_a = np.array([x[2] for x in tanimotos_l])
        ev = ((tanimotos_a <= 0.8).sum() == tanimotos_a.size).astype(int)
        if ev != 1:
            #cluster_to_evaluate.append((cluster, tanimotos_l))
            
            with open('clusts_out/'+in_file+'/'+cluster_name+'.csv', 'w') as fopen:
                fopen.write(str(cluster))
                fopen.write('\n')
                for i in tanimotos_l:
                    fopen.write(str(i[0])+';'+str(i[1])+';'+str(i[2])+'\n')
            fopen.close()
    '''        
    with open('clusts_out/'+sys.argv[1]+'.out', 'w') as fopen:
        fopen.write('Clusters in '+sys.argv[1]+' that did not match specifications.\n')
        for clust in cluster_to_evaluate:
            fopen.write(str(clust))
            fopen.write('\n')
    fopen.close()
    '''
        
