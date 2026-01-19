

import argparse

from joblib import Parallel, delayed
from water_completion_methods.nearby_atoms import get_predicted_ligand_waters
from water_completion_methods.findwaters import findwaters, findwaters_multiple
import pandas as pd
import numpy as np
import gemmi
from pathlib import Path
import yaml

def output_input_yaml(hits, out_path):
    output_order = np.random.permutation([x for x in hits.keys()])
    print(output_order)

    input_yaml = {
        int(j): hits[int(j)]
        for j
        in output_order
    }

    with open(out_path, 'w') as f:
        yaml.dump(input_yaml, f, )

def main(data_path, out_path):
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)

    output_input_yaml(data, out_path)

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
