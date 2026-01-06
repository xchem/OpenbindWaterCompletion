import gemmi 

def base_structure_waters(structure, xmap, out_dir, chain, res, ):
    """
    Return the waters in the base structure around the ligand with no changes.
    """
    st = gemmi.read_structure(str(structure))

    # Get the water atoms
    water_atoms = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == 'HOH':
                    for atom in res:
                        pos = atom.pos
                        # water_atoms[(chain.name, str(res.seqid.num))] = 
                        water_atoms.append([pos.x, pos.y, pos.z])

    return water_atoms