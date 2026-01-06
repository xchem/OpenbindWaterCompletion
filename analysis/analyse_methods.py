import pathlib

import numpy as np
import gemmi
from joblib import Parallel, delayed

from water_completion_methods.base_structure_waters import base_structure_waters
from water_completion_methods.findwaters import findwaters, findwaters_multiple
from water_completion_methods.nearby_atoms import get_ligand_waters

def get_water_data(data_path, data):
    water_data = {}
    for (system, dtag, chain, res), paths in data.items():
        st = gemmi.read_structure(str(data_path / system / dtag / paths['bound']))
        water_data[(system, dtag, chain, res)] = (
            data_path / system / dtag / paths['ground'], 
                        data_path / system / dtag / paths['bound'], 
            data_path / system / dtag / paths['map'], 
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
            if ligand_results['num_waters'] != 0.0:
                ligand_precision = sum(water_classes.values()) / ligand_results['num_waters']
            else:
                ligand_precision = 0.0
            print(f'\t# {ligand[0]} {ligand[1]} {ligand[2]} {ligand[3]}')
            num_waters = ligand_results["num_waters"]
            print(f'\t\tRecall: {round(ligand_recall, 2)}')
            print(f'\t\tPrecision: {round(ligand_precision, 2)}')
            print(f'\t\tNumber of water predictions: {round(num_waters, 2)}')

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

def process_dataset(method, bound_structure, xmap, out_dir, chain, res, waters):
    predicted_waters = method(bound_structure, xmap, out_dir, chain, res, )

        

    predicted_ligand_waters = get_predicted_ligand_waters(
                bound_structure, 
                predicted_waters,
                chain,
                res,
                )
    result_analysis = analyse_result(waters, predicted_ligand_waters,)
    return result_analysis

def analyse_methods(methods, data, data_path):

    # Get the results for each method
    all_results = {}
    for method_name, method in methods.items():
        print(f'Processing method: {method_name}...')
        all_results[method_name] = {}
        predicted_waters_futures = {}
        for (system, dtag, chain, res), (structure, bound_structure, xmap, waters) in data.items():
            out_dir = data_path / system / dtag 
            predicted_waters_futures[(system, dtag, chain, res)] = delayed(process_dataset)(
                method, bound_structure, xmap, out_dir, chain, res, waters)

        results = Parallel(n_jobs=-1)(f for f in predicted_waters_futures.values())
        for result_id, result in zip(data, results):
            all_results[method_name][result_id] = result



    # Summarize the results
    summarize_results(all_results)

if __name__ == "__main__":
    ...

    methods = {
        'base': base_structure_waters,
        'findwaters_sigma_2.0': lambda st, xmap, chain, res, out_dir: findwaters(st, xmap, chain, res, out_dir, sigma=2.0),
        'findwaters_sigma_3.0': lambda st, xmap, chain, res, out_dir: findwaters(st, xmap, chain, res, out_dir, sigma=3.0),
        'findwaters_sigma_4.0': lambda st, xmap, chain, res, out_dir: findwaters(st, xmap, chain, res, out_dir, sigma=4.0),
        'findwaters_sigma_5.0': lambda st, xmap, chain, res, out_dir: findwaters(st, xmap, chain, res, out_dir, sigma=5.0),
        'findwaters_multiple_2_steps': lambda st, xmap, chain, res, out_dir: findwaters_multiple(st, xmap, chain, res, out_dir, sigmas=np.linspace(5.0,0.5,num=2)),
        'findwaters_multiple_3_steps': lambda st, xmap, chain, res, out_dir: findwaters_multiple(st, xmap, chain, res, out_dir, sigmas=np.linspace(5.0,0.5,num=3)),
        'findwaters_multiple_5_steps': lambda st, xmap, chain, res, out_dir: findwaters_multiple(st, xmap, chain, res, out_dir, sigmas=np.linspace(5.0,0.5,num=5)),
        'findwaters_multiple_7_steps': lambda st, xmap, chain, res, out_dir: findwaters_multiple(st, xmap, chain, res, out_dir, sigmas=np.linspace(5.0,0.5,num=7)),
        'findwaters_multiple_9_steps': lambda st, xmap, chain, res, out_dir: findwaters_multiple(st, xmap, chain, res, out_dir, sigmas=np.linspace(5.0,0.5,num=9)),
        'findwaters_multiple_11_steps': lambda st, xmap, chain, res, out_dir: findwaters_multiple(st, xmap, chain, res, out_dir, sigmas=np.linspace(5.0,0.5,num=11)),

        # 'findwaters_multiple_10_steps': lambda st, xmap, chain, res: findwaters_multiple(st, xmap, chain, res, sigmas=np.linspace(4.0,0.5,num=10)),
        # 'findwaters_multiple_17_steps': lambda st, xmap, chain, res: findwaters_multiple(st, xmap, chain, res, sigmas=np.linspace(4.0,0.5,num=17)),

        # 'findwaters_multiple_exhaustive': lambda st, xmap, chain, res: findwaters_multiple(st, xmap, chain, res, sigmas=np.linspace(4.0,0.5,num=34)),

        # '': ...,
    }
    data = {
        ('CHIKV', 'CHIKV_MacB-x0270', 'A', '304'): {
            'ground': 'CHIKV_MacB-x0270-pandda-input.pdb',
            'bound': 'refine.split.bound-state.pdb',
            'map': 'CHIKV_MacB-x0270-event_1_1-BDC_0.14_map.native.ccp4'
        },
        ('NXT1', 'x5052a', 'B', '303'): {
            'ground': 'x5052a.pdb',
            'bound': 'x5052a.pdb',
            'map': 'x5052a_event_crystallographic.ccp4'
        },
        ('NXT1', 'x5080a', 'B', '303'): {
            'ground': 'x5080a.pdb',
            'bound': 'x5080a.pdb',
            'map': 'x5080a_event_crystallographic.ccp4'
        },
        ('A71EV2A', 'A0152b', 'A', '201'): {
            'ground': 'A0152b.pdb',
            'bound': 'A0152b.pdb',
            'map': 'A0152b_event_crystallographic.ccp4'
        },
        ('A71EV2A', 'A0194a', 'A', '147'): {
            'ground': 'A0194a.pdb',
            'bound': 'A0194a.pdb',
            'map': 'A0194a_event_crystallographic.ccp4'
        },
        ('A71EV2A', 'A0202a', 'A', '147'): {
            'ground': 'A0202a.pdb',
            'bound': 'A0202a.pdb',
            'map': 'A0202a_event_crystallographic.ccp4'
        },        
        ('A71EV2A', 'A0207a', 'A', '151'): {
            'ground': 'A0207a.pdb',
            'bound': 'A0207a.pdb',
            'map': 'A0207a_event_crystallographic.ccp4'
        },
        ('A71EV2A', 'A0228a', 'A', '147'): {
            'ground': 'A0228a.pdb',
            'bound': 'A0228a.pdb',
            'map': 'A0228a_event_crystallographic.ccp4'
        },
        ('A71EV2A', 'A0237a', 'A', '151'): {
            'ground': 'A0237a.pdb',
            'bound': 'A0237a.pdb',
            'map': 'A0237a_event_crystallographic.ccp4'
        },
        ('A71EV2A', 'A0310a', 'A', '147'): {
            'ground': 'A0310a.pdb',
            'bound': 'A0310a.pdb',
            'map': 'A0310a_event_crystallographic.ccp4'
        },
        ('A71EV2A', 'A0526a', 'A', '147'): {
            'ground': 'A0526a.pdb',
            'bound': 'A0526a.pdb',
            'map': 'A0526a_event_crystallographic.ccp4'
        },

    }

    data_path = pathlib.Path('./data')

    water_data = get_water_data(data_path, data)

    analyse_methods(methods, water_data, data_path)