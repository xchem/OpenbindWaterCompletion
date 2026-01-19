import argparse

from joblib import Parallel, delayed
from water_completion_methods.nearby_atoms import get_predicted_ligand_waters
from water_completion_methods.findwaters import findwaters, findwaters_multiple
import pandas as pd
import numpy as np
import gemmi
from pathlib import Path
import yaml


def get_high_confidence_hits(data_path):
    high_confidence_hits = []
    for pandda_dir in data_path.glob('*'):
        inspect_table_path = pandda_dir / 'analyses' / 'pandda_inspect_events.csv'
        if not inspect_table_path.exists():
            print(f'Skipping dir: {pandda_dir}')
            continue
        inspect_table = pd.read_csv(inspect_table_path)
        for idx, row in inspect_table.iterrows():
            dtag, event_idx, bdc, x, y, z, confidence = row['dtag'], row['event_idx'], row['1-BDC'],row['x'],row['y'],row['z'], row['Ligand Confidence'] 
            if confidence != 'High':
                continue
            dataset_dir = pandda_dir / 'processed_datasets' / dtag

            high_confidence_hits.append(
                {
                    'System': str(pandda_dir.name),
                    'Dtag': str(dtag),
                    'EventIdx': int(event_idx),
                    'DatasetDir': str(dataset_dir),
                    'xyz': [x, y, z],
                    'InitialStructurePath': str(dataset_dir / f'{dtag}-pandda-input.pdb'),
                    'BoundStatePath': str(dataset_dir / 'modelled_structures' / f'{dtag}-pandda-model.pdb'),
                    'WaterBuildPath': str(dataset_dir / f'findwaters_multiple_21.pdb'),
                    'EventMapPath': str(dataset_dir / f"{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.ccp4"),
                    'EventMTZPath': str(dataset_dir / f"{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.mtz"), 
                }
            )

    return high_confidence_hits

def get_mean_pos(res):
    poss = []
    for atom in res:
        pos = atom.pos
        poss.append([pos.x, pos.y, pos.z])

    return np.mean(poss, axis=0)

def get_closest_ligand(bound_state_path, xyz):
    st = gemmi.read_structure(str(bound_state_path))

    distances = {}
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == "LIG":
                    mean_pos = get_mean_pos(res)
                    if mean_pos.size != 3:
                        raise Exception(f'Mean pos array size should be 3, is {mean_pos.size}')
                    distances[(chain.name, res.seqid.num)] = np.linalg.norm(
                        np.array(
                            [
                                xyz[0]-mean_pos[0],
                                xyz[1]-mean_pos[1],
                                xyz[2]-mean_pos[2],
                                
                            ]
                        )
                    )

    if len(distances) == 0:
        return None, None
    else:
        return min(distances, key=lambda _x: distances[_x])

    ...

def make_event_mtz(event_map_path, structure_pdb, output_path):
    ccp4 = gemmi.read_ccp4_map(str(event_map_path))
    ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')
    ccp4.setup(0.0)
    arr = np.array(ccp4.grid, copy=False)
    arr[np.isnan(arr)] = 0.0
    st = gemmi.read_structure(str(structure_pdb))


    sf = gemmi.transform_map_to_f_phi(ccp4.grid, half_l=True)
    data = sf.prepare_asu_data(st.resolution)
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = sf.spacegroup
    mtz.set_cell_for_all(sf.unit_cell)
    mtz.add_dataset('unknown')
    mtz.add_column('FWT', 'F')
    mtz.add_column('PHWT', 'P')
    mtz.set_data(data)

    mtz.write_to_file(str(output_path))


def process_hit(hit):
    print(f'Processing Hit: {hit["Dtag"]} {hit["EventIdx"]}')
    try:
        # Get an mtz from the waters and apply findwaters_multiple

        # Get the chain and res of closest ligand
        chain, res = get_closest_ligand(hit['BoundStatePath'], hit['xyz'])
        if not chain:
            return None
        print(chain, res)

        # Model structure waters
        waters = findwaters_multiple(
            hit['BoundStatePath'], 
            hit['EventMapPath'], 
            Path(hit['DatasetDir']), 
            chain, 
            res, 
            sigmas=np.geomspace(5.0,0.5,num=21))

        # Make the mtz
        make_event_mtz(
            hit['EventMapPath'],
            hit['InitialStructurePath'],
            hit['EventMTZPath'],
        )
        predicted_ligand_waters = get_predicted_ligand_waters(
                hit['BoundStatePath'], 
                waters,
                chain,
                res,
                )
        print(predicted_ligand_waters)

        return {
            'dtag': hit['Dtag'],
            'pdb': hit['WaterBuildPath'],
            'xmap': hit['EventMTZPath'],
            'landmarks': {
            j+1: [water[0], water[1]] for j, water in enumerate(predicted_ligand_waters)
            }
        }
    except Exception as e:
        print(e)
        return None
        ...


def output_input_yaml(hits, out_path):
    output_order = np.random.permutation(len(hits))

    input_yaml = {
        int(j)+1: hits[int(j)]
        for j
        in output_order
    }

    with open(out_path, 'w') as f:
        yaml.dump(input_yaml, f, )


def main(data_path, out_path):
    # Iterate over PanDDAs

    # Get the high confidence hits, with paths to relevant data for water building
    high_confidence_hits = get_high_confidence_hits(data_path)
    # print(high_confidence_hits)

    # Dispatch jobs to do water fitting and create mtzs
    futures = []
    for high_confidence_hit in high_confidence_hits:
        futures.append(
            delayed(process_hit)(high_confidence_hit)
        )
    results = Parallel(n_jobs=-1, verbose=50)(f for f in futures)
    print(results)
    succesful_results = [r for r in results if r]
    print(f'Got {len(succesful_results)} out of {len(results)} jobs')

    # Create a input yaml for annotation
    output_input_yaml(succesful_results, out_path)


    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--data_path')
    parser.add_argument('--out_path')
    args=parser.parse_args()
    print(f'Data Path: {args.data_path}')
    print(f'Out Path: {args.out_path}')
    main(Path(args.data_path), Path(args.out_path))
