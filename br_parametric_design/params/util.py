import numpy as np
import biotite.structure as bio_struct


def _construct_param_vector(param_dict: dict, param_names: list):
    """
    Construct a parameter vector from the parameter dictionary based on the specified names.

    Parameters
    ----------
    param_dict : dict
        The dictionary containing parameter names and their values which will be picked by param_names.
    param_names : list
        The list of parameter names to include in the vector. Order of names determines order in vector.

    Returns
    -------
    param_vector : np.ndarray
        The constructed parameter vector.
    parse_dict : dict
        A dictionary mapping parameter names to their indices or slices in the param_vector.
    """
    param_vector = []
    for name in param_names:
        if isinstance(param_dict[name], (int, float)):
            param_vector.append(param_dict[name])
        else:
            param_vector.extend(param_dict[name])
    param_vector = np.array(param_vector)

    parse_dict = {}
    param_index = 0
    for name in param_names:
        if isinstance(param_dict[name], (int, float)):
            parse_dict[name] = param_index
            param_index += 1
        else:
            parse_dict[name] = slice(param_index, param_index + len(param_dict[name]))
            param_index += len(param_dict[name])

    return param_vector, parse_dict


def _construct_param_dict(params: np.ndarray, parse_dict: dict):
    """
    Construct a parameter dictionary from the parameter vector based on the parse dictionary.

    Parameters
    ----------
    params : np.ndarray
        The flattened parameter vector.
    parse_dict : dict
        A dictionary mapping parameter names to their indices or slices in the params vector.

    Returns
    -------
    param_dict : dict
        The constructed parameter dictionary.
    """
    param_dict = {}
    for key, value in parse_dict.items():
        param_dict[key] = params[value]
    return param_dict


def ca_xyz_to_atom_array(xyz, chain_id_i="A", res_name="GLY"):

    if len(xyz.shape) == 3:
        helix_len = xyz.shape[1]
        length = xyz.shape[0] * xyz.shape[1]
    elif len(xyz.shape) == 2:
        helix_len = xyz.shape[0]
        length = xyz.shape[0]
    else:
        raise ValueError(
            "xyz must be of shape (helix_num, residue_num, 3) or (residue_num, 3)"
        )
    xyz = np.reshape(xyz, (-1, 3))
    structure = bio_struct.AtomArray(length=length)
    structure.atom_name = np.array(["CA"] * length)
    structure.chain_id = np.array(
        [chr(ord(chain_id_i) + i // helix_len) for i in range(length)]
    )
    structure.res_id = np.array(list(range(1, helix_len + 1)) * (length // helix_len))
    structure.element = np.array(["C"] * length)
    structure.res_name = np.array([res_name] * length)
    structure.coord = xyz
    return structure
