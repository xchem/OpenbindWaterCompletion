import subprocess
import os 

import gemmi 

from water_completion_methods.nearby_atoms import get_ligand_waters

SCRIPT = "module load ccp4; findwaters --pdbin {pdb_in_filename} --mapin {map_file} --pdbout {waters_filename} --sigma {sigma_level} --min-dist {min_dist_to_protein} --max-dist {max_dist_to_protein}"

def remove_waters(st):
    st.clone().remove_waters()

    return st
    
def get_waters(st_desolv, waters_pdb,  chain, res,):
    water_st = gemmi.read_structure(waters_pdb)

    water_st[0].add_chain(st_desolv[0][chain])

    ligand_waters = get_ligand_waters(chain, res, water_st, )

    return [x for x in ligand_waters.values()]
    

def findwaters(structure, xmap, chain, res, sigma=2.0, min_dist=1.4, max_dist=7.0):
    """
    Return the waters in the base structure around the ligand with no changes.
    """
    st = gemmi.read_structure(str(structure))

    # Clear the waters
    st_desolv = remove_waters(st)

    # Output the transformed file
    desolv_pdb = 'desolv.pdb'
    st_desolv.write_minimal_pdb(desolv_pdb)

    # Run findwaters
    waters_pdb = 'waters.pdb'
    p = subprocess.Popen(
        SCRIPT.format(
            pdb_in_filename=desolv_pdb, 
            map_file=xmap,
            waters_filename=waters_pdb, 
            sigma_level=sigma,
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
    os.remove(waters_pdb)
    os.remove(desolv_pdb)

    return waters


    ...
