import gemmi
import numpy as np

def get_nearby_atoms(reference, moving, threshold=7.0):
    reference_atom_array = np.array(reference)
    moving_atom_array = np.array(moving)

    distance_matrix = np.linalg.norm((reference_atom_array.reshape(1,-1,3)-moving.reshape(-1,1,3)), axis=2)

    close = []
    for water_atom_id, distances in zip(moving, distance_matrix):
        min_dist = min(distances)
        if min_dist < threshold:
            close.append(True)
        else:
            close.append(False)
    return close

def get_nearby_atoms_from_gemmi(reference, moving, threshold=7.0):
    reference_poss = [(pos.x,pos.y,pos.z) for pos in reference]
    moving_poss = [(pos.x,pos.y,pos.z) for pos in moving]
    return get_nearby_atoms(reference_poss, moving_poss, threshold)


def get_ligand_waters(chain, res, st, threshold=7.0):
    """
    Get waters near the ligand
    """
    # st = gemmi.read_structure(str(bound_state_path))

    print(st)
    print([chain, res])
    ligand_res = st[0][chain][res][0]

    # Get the ligand atoms
    ligand_atoms = []
    for atom in ligand_res:
        pos = atom.pos 
        ligand_atoms.append([pos.x, pos.y, pos.z])
    ligand_atom_array = np.array(ligand_atoms)

    # Get the water atoms
    water_atoms = {}
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == 'HOH':
                    for atom in res:
                            pos = atom.pos
                            water_atoms[(chain.name, str(res.seqid.num))] = [pos.x, pos.y, pos.z]

    water_atom_array = np.array([x for x in water_atoms.values()])
    # print(ligand_atom_array)
    # print(water_atom_array)

    distance_matrix = np.linalg.norm((ligand_atom_array.reshape(1,-1,3)-water_atom_array.reshape(-1,1,3)), axis=2)

    ligand_waters = {}
    for water_atom_id, distances in zip(water_atoms, distance_matrix):
        min_dist = min(distances)
        if min_dist < threshold:
            ligand_waters[water_atom_id] = water_atoms[water_atom_id]

    # print(ligand_waters)

    return ligand_waters