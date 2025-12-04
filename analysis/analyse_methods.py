import pathlib

import numpy as np
import gemmi

from water_completion_methods.base_structure_waters import base_structure_waters
from water_completion_methods.findwaters import findwaters, findwaters_multiple
from water_completion_methods.nearby_atoms import get_ligand_waters

def get_water_data(data_path, data):
    water_data = {}
    for (dtag, chain, res), paths in data.items():
        st = gemmi.read_structure(str(data_path / dtag / paths['bound']))
        water_data[(dtag, chain, res)] = (
            data_path / dtag / paths['ground'], 
                        data_path / dtag / paths['bound'], 
            data_path / dtag / paths['map'], 
            get_ligand_waters(chain, res, st),
            )
    
    return water_data

def analyse_result(waters, predicted_waters, threshold=0.5):
    """
    Iterate over known waters, getting the distance to the closest predicted water
    Classify as positive/negative on whether they are within 0.5A
    """
    # 
    closest_distances = {}
    water_classes = {}
    unmatched_waters = []
    for (chain, res), (x,y,z) in waters.items():
        distances = [
            np.linalg.norm([x-x_mov, y-y_mov, z-z_mov]) 
            for j, (x_mov, y_mov, z_mov) 
            in predicted_waters.items()
        ]
        if len(distances) == 0:
            closest_water_distance = 10000.0
        else:
            closest_water_distance = min(distances)
        closest_distances[(chain, res)] = closest_water_distance
        if closest_water_distance < threshold:
            water_classes[(chain, res)] = 1
        else:
            water_classes[(chain, res)] = 0
            unmatched_waters.append((chain,res))

    print(unmatched_waters)

    print(sorted(closest_distances.values()))
    return {
        'closest_water_distances': closest_distances,
        'water_classes': water_classes,
        'num_waters': len(predicted_waters)
    }

def summarize_results(all_results):
    for method, method_results in all_results.items():
        print(f'# Method: {method}')
        for ligand, ligand_results in method_results.items():
            closest_water_distances = ligand_results['closest_water_distances']
            water_classes = ligand_results['water_classes']
            if len(water_classes) != 0.0:
                ligand_recall = sum(water_classes.values()) / len(water_classes)
            else: 
                ligand_recall = 0.0
            if len(ligand_results['num_waters']) != 0.0:
                ligand_precision = sum(water_classes.values()) / ligand_results['num_waters']
            else:
                ligand_precision = 0.0
            print(f'\t# {ligand[0]} {ligand[1]} {ligand[2]}')
            num_waters = ligand_results["num_waters"]
            print(f'\t\tRecall: {ligand_recall}')
            print(f'\t\tPrecision: {ligand_precision}')
            print(f'\t\tNumber of water predictions: {num_waters}')

    ...

def get_predicted_ligand_waters(
                bound_state_path, 
                predicted_ligand_waters,
                chain,
                res,
                threshold=7.0,
                ):
    st = gemmi.read_structure(str(bound_state_path))

    ligand_res = st[0][chain][res][0]

    # Get the ligand atoms
    ligand_atoms = []
    for atom in ligand_res:
        pos = atom.pos 
        ligand_atoms.append([pos.x, pos.y, pos.z])
    ligand_atom_array = np.array(ligand_atoms)

    #
    water_atoms = {str(j): pos for j, pos in enumerate(predicted_ligand_waters)}
    water_atom_array = np.array(predicted_ligand_waters)

    #
    distance_matrix = np.linalg.norm((ligand_atom_array.reshape(1,-1,3)-water_atom_array.reshape(-1,1,3)), axis=2)

    ligand_waters = {}
    for water_atom_id, distances in zip(water_atoms, distance_matrix):
        min_dist = min(distances)
        if min_dist < threshold:
            ligand_waters[water_atom_id] = water_atoms[water_atom_id]

    return ligand_waters

def analyse_methods(methods, data):

    # Get the results for each method
    all_results = {}
    for method_name, method in methods.items():
        all_results[method_name] = {}
        for (dtag, chain, res), (structure, bound_structure, xmap, waters) in data.items():
            predicted_waters = method(structure, xmap, chain, res)
            predicted_ligand_waters = get_predicted_ligand_waters(
                bound_structure, 
                predicted_waters,
                chain,
                res,
                )
            result_analysis = analyse_result(waters, predicted_ligand_waters,)
            all_results[method_name][(dtag, chain, res)] = result_analysis

    # Summarize the results
    summarize_results(all_results)

if __name__ == "__main__":
    ...

    methods = {
        'base': base_structure_waters,
        'findwaters_sigma_2.0': lambda st, xmap, chain, res: findwaters(st, xmap, chain, res, sigma=2.0),
        'findwaters_sigma_3.0': lambda st, xmap, chain, res: findwaters(st, xmap, chain, res, sigma=3.0),
        'findwaters_sigma_4.0': lambda st, xmap, chain, res: findwaters(st, xmap, chain, res, sigma=4.0),
        'findwaters_sigma_5.0': lambda st, xmap, chain, res: findwaters(st, xmap, chain, res, sigma=5.0),
        'findwaters_multiple_minimal': lambda st, xmap, chain, res: findwaters_multiple(st, xmap, chain, res, sigmas=np.linspace(4.0,0.5,num=17)),
        'findwaters_multiple_exhaustive': lambda st, xmap, chain, res: findwaters_multiple(st, xmap, chain, res, sigmas=np.linspace(4.0,0.5,num=34)),

        # '': ...,
    }
    data = {
        ('CHIKV_MacB-x0270', 'A', '304'): {
            'ground': 'CHIKV_MacB-x0270-pandda-input.pdb',
            'bound': 'refine.split.bound-state.pdb',
            'map': 'CHIKV_MacB-x0270-event_1_1-BDC_0.14_map.native.ccp4'
        }
    }

    data_path = pathlib.Path('./data')

    water_data = get_water_data(data_path, data)

    analyse_methods(methods, water_data)