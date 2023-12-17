from utils import *
from agent import RDmasterAgent


data_folder = 'dataset'
dise_dict = text_to_dict('{}/all_orpha.txt'.format(data_folder))  # all ORPHA disease
phe_dict = text_to_dict('{}/all_orpha_phe_hpo.txt'.format(data_folder))  # all phenotypes related to ORPHA, including ANC
phe_plus_dict = text_to_dict('{}/all_phe_hpo.txt'.format(data_folder))  # all PHENOTYPIC_ABNORMALITY phenotypes
dise_anno_phe = sparse.load_npz('{}/orpha2hpo.npz'.format(data_folder))  # original annotated relation between ORPHA and phenotypes
dise_phe_pro = sparse.load_npz('{}/orpha_hpo_pro.npz'.format(data_folder))  # phenotype freq/probability of each ORPHA, including ANC
dise_phe_max_pro = sparse.load_npz('{}/orpha_hpo_max_pro.npz'.format(data_folder))  # max phenotype freq/probability of each ORPHA, including ANC
phe_dise_pro = sparse.load_npz('{}/hpo_orpha_pro.npz'.format(data_folder))  # distribution probability of each phenotype among all ORPHA, considering ANC
dise_dise_pro = sparse.load_npz('{}/dise_dise_pro.npz'.format(data_folder))
phe_phe_pro = sparse.load_npz('{}/hpo_hpo_pro.npz'.format(data_folder))  # hpo co_occurrence conditional probability
phe_background_freq = np.loadtxt('{}/hpo_background_freq.txt'.format(data_folder))  # phenotype background frequency score
phe_entropy = np.loadtxt('{}/hpo_entropy.txt'.format(data_folder))  # phenotype background entropy
phe_gini = np.loadtxt('{}/hpo_gini.txt'.format(data_folder))  # phenotype background gini
phe_anc_dict, phe_parent_dict, phe_desc_dict, phe_child_dict = get_anc_and_desc_dict(
    sparse.load_npz('{}/all_phe_hpo_anc.npz'.format(data_folder)).toarray(), phe_plus_dict)  # phenotype ancestor and descendant relation matrix
observed_lr = sparse.load_npz('{}/orpha_hpo_lr_observed.npz'.format(data_folder))
excluded_lr = sparse.load_npz('{}/orpha_hpo_lr_excluded.npz'.format(data_folder))

agent = RDmasterAgent(phe_dict, phe_plus_dict, dise_dict, dise_anno_phe, dise_phe_pro, dise_phe_max_pro, phe_dise_pro, phe_phe_pro, dise_dise_pro, phe_background_freq, phe_entropy,
                      phe_gini, phe_anc_dict, phe_desc_dict, observed_lr, excluded_lr)
# give a HPO list
agent.next([], [], [])
