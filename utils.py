import numpy as np
from scipy import sparse


def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices """
    keys = np.loadtxt(path, dtype=str)
    indices = np.arange(len(keys))
    return dict(zip(keys, indices))


def get_anc_and_desc_dict(all_phe_hpo_anc, phe_plus_dict):
    phe_anc_mat = sparse.csr_matrix(all_phe_hpo_anc)
    phe_desc_mat = sparse.csc_matrix(all_phe_hpo_anc)

    phe_anc_dict = {}
    phe_parent_dict = {}
    for phe, phe_idx in phe_plus_dict.items():
        phe_anc_dict[phe] = list(phe_anc_mat[phe_idx].indices)
        phe_parent_dict[phe] = list(phe_anc_mat[phe_idx].indices[np.where(phe_anc_mat[phe_idx].data == 1)])

    phe_desc_dict = {}
    phe_child_dict = {}
    for phe, phe_idx in phe_plus_dict.items():
        phe_desc_dict[phe] = list(phe_desc_mat[:, phe_idx].indices)
        phe_child_dict[phe] = list(phe_desc_mat[:, phe_idx].indices[np.where(phe_desc_mat[:, phe_idx].data == 1)])

    return phe_anc_dict, phe_parent_dict, phe_desc_dict, phe_child_dict
