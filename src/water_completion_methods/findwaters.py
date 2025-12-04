import subprocess
import os 
import time 

import gemmi 
import numpy as np

from water_completion_methods.nearby_atoms import get_ligand_waters

SCRIPT = "module load ccp4; findwaters --pdbin {pdb_in_filename} --mapin {map_file} --pdbout {waters_filename} --sigma {sigma_level} --min-dist {min_dist_to_protein} --max-dist {max_dist_to_protein}"

def remove_waters(st):
    new_st = st.clone()
    new_st.remove_waters()

    return new_st
    
def get_waters(st, waters_pdb,  chain, res,):
    water_st = gemmi.read_structure(waters_pdb)

    # water_st[0].add_chain(st[0][chain])

    # ligand_waters = get_ligand_waters(chain, res, water_st, )

    # return [x for x in ligand_waters.values()]
    waters = []
    for _model in water_st:
        for _chain in _model:
            for _res in _chain:
                for _atom in _res:
                    pos = _atom.pos
                    waters.append([pos.x, pos.y, pos.z])
    
    return waters

def map_sigma(xmap, sigma):
    ccp4 = gemmi.read_ccp4_map(str(xmap))
    grid = ccp4.grid
    grid_array = np.array(grid, copy=False)
    std = np.std(grid_array)
    non_zero_std = np.std(grid_array[grid_array != 0.0])

    print(f'Stds : with zero vs without: {std} / {non_zero_std}. New Sigma: {round(float(new_sigma), 2)}')
    new_sigma = sigma * (non_zero_std / std)
    return new_sigma

def findwaters(structure, xmap, chain, res, sigma=2.0, min_dist=1.4, max_dist=7.0):
    """
    Return the waters in the base structure around the ligand with no changes.
    """
    st = gemmi.read_structure(str(structure))

    # Clear the waters
    st_desolv = remove_waters(st)

    # Map sigma
    new_sigma = map_sigma(xmap, sigma)

    # Output the transformed file
    desolv_pdb = f'desolv.pdb'
    st_desolv.write_minimal_pdb(desolv_pdb)

    # Run findwaters
    waters_pdb = f'waters_{round(float(sigma), 2)}.pdb'
    p = subprocess.Popen(
        SCRIPT.format(
            pdb_in_filename=desolv_pdb, 
            map_file=xmap,
            waters_filename=waters_pdb, 
            sigma_level=new_sigma,
            min_dist_to_protein=min_dist, 
            max_dist_to_protein=max_dist
        ),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
    stdout, stderr = p.communicate()
    print(f'STDOUT: {stdout}')
    print(f'STDERR: {stderr}')

    # Get ligand waters from output file
    waters = get_waters(st, waters_pdb, chain, res,)

    # Cleanup
    # os.remove(waters_pdb)
    # os.remove(desolv_pdb)

    print(f'Got {len(waters)} waters')
    return waters


    ...


def findwaters_multiple(structure, xmap, chain, res, sigmas=np.linspace(10.0,0.7,num=93), min_dist=1.4, max_dist=7.0):
    waters = []
    for sigma in sigmas:
        waters += findwaters(structure,xmap, chain, res, sigma=sigma)

    return waters