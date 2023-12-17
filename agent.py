import numpy as np
from scipy import sparse
from diagnosis.hpo_selector import HpoSelector


class RDmasterAgent:
    DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR = 0.01

    def __init__(self, phe_dict=None, phe_plus_dict=None, dise_dict=None, dise_anno_phe=None, dise_phe_pro=None, dise_phe_max_pro=None, phe_dise_pro=None,
                 phe_phe_pro=None, dise_dise_pro=None,
                 phe_background_freq=None, phe_entropy=None, phe_gini=None, phe_anc_dict=None, phe_desc_dict=None, observed_lr=None,
                 excluded_lr=None):

        self.phe_dict = phe_dict
        self.phe_plus_dict = phe_plus_dict
        self.all_orpha_hpo = list(self.phe_dict.keys())
        self.all_phe_hpo = list(self.phe_plus_dict.keys())
        self.dise_dict = dise_dict
        self.all_dise = list(self.dise_dict.keys())
        self.dise_num = len(dise_dict)
        self.dise_anno_phe = dise_anno_phe
        self.phe_dise_pro = phe_dise_pro
        self.dise_phe_pro = dise_phe_pro
        self.dise_phe_pro_T = sparse.csr_matrix(self.dise_phe_pro.toarray().T)
        self.dise_phe_pro_sum = dise_phe_pro.sum(axis=1).A.reshape(1, -1)[0]
        self.dise_phe_max_pro = dise_phe_max_pro
        self.phe_phe_pro = phe_phe_pro
        self.dise_dise_pro = dise_dise_pro
        self.phe_background_freq = phe_background_freq
        self.phe_entropy = phe_entropy.reshape(1, -1)
        self.phe_gini = phe_gini.reshape(1, -1)
        self.phe_anc_dict = phe_anc_dict
        self.phe_desc_dict = phe_desc_dict
        self.observed_lr = observed_lr
        self.excluded_lr = excluded_lr

        self.default_phe_background_freq = 1 / len(self.phe_plus_dict)
        self.pretest_odds = 1 / len(self.dise_dict)
        self.observed_lr = self.lr_sparse_matrix_to_dict(self.observed_lr)
        self.excluded_lr = self.lr_sparse_matrix_to_dict(self.excluded_lr)

        self.phe_hpo_idx_to_orpha_hpo_idx = np.full(len(self.phe_plus_dict), -1, dtype=int)
        self.orpha_hpo_idx_to_phe_hpo_idx = np.full(len(self.phe_dict), -1, dtype=int)
        self.hpo_idx_transfer()

        self.hpo_selector = HpoSelector(
            dise_dict=self.dise_dict,
            phe_dict=self.phe_dict,
            dise_phe_max_pro=self.dise_phe_max_pro
        )

        # self.dise_pro_calculator = DiseaseLikelihoodRatio(
        #     all_orpha_hpo=self.phe_dict,
        #     all_phe_hpo=self.phe_plus_dict,
        #     all_orpha=self.dise_dict,
        #     hpo_anc_mat=self.phe_anc_dict,
        #     hpo_desc_mat=self.phe_desc_dict,
        #     hpo_background_freq=self.phe_background_freq,
        #     orpha2hpo=self.dise_anno_phe,
        #     orpha_hpo_pro=self.dise_phe_pro,
        #     orpha_hpo_max_pro=self.dise_phe_max_pro
        # )

    def next(self, positive_phe_list, negative_phe_list, not_sure_phe_list):
        """the method to get the interrogated phenotype"""
        phe_flag = np.ones(len(self.phe_dict))  # agent will not ask unrelated phenotypes
        for slot in positive_phe_list + negative_phe_list + not_sure_phe_list:
            # set the informed phenotype flag to 0
            idx = self.phe_dict.get(slot, -1)
            if idx != -1: phe_flag[idx] = 0
        # filter positive anc
        anc_phe_idx = set()
        for phe in positive_phe_list:
            anc_idx = [self.phe_hpo_idx_to_orpha_hpo_idx[idx] for idx in self.phe_anc_dict.get(phe)]
            anc_phe_idx.update(anc_idx)
        anc_phe_idx.discard(-1)
        if len(anc_phe_idx) > 0: phe_flag[np.array(list(anc_phe_idx))] = 0
        # filter negative desc
        desc_phe_idx = set()
        for phe in negative_phe_list:
            desc_idx = [self.phe_hpo_idx_to_orpha_hpo_idx[idx] for idx in self.phe_desc_dict.get(phe)]
            desc_phe_idx.update(desc_idx)
        desc_phe_idx.discard(-1)
        if len(desc_phe_idx) > 0: phe_flag[np.array(list(desc_phe_idx))] = 0

        disease_prob = self.evaluate_disease_prob_by_phenotype(positive_phe_list, negative_phe_list)

        self.hpo_selector.set_class_weight(disease_prob)
        cur_gini, phe_gini_prior_score, phe_info_prior_score = self.hpo_selector.calculate_gini_gain_and_info_gain(get_gini_gain=True, get_info_gain=True)

        return np.argmax(phe_flag * ((1-cur_gini)*phe_gini_prior_score + cur_gini*phe_info_prior_score))

    def evaluate_disease_prob_by_phenotype(self, positive_phe_list, negative_phe_list):
        disease_posterior_probability = np.zeros(len(self.dise_dict))
        # hooks
        get_phe_idx = self.phe_plus_dict.get
        get_orpha_observed = self.observed_lr.get
        get_orpha_excluded = self.observed_lr.get
        observed_lr = []
        append_observed_lr = observed_lr.append
        clear_observed_lr = observed_lr.clear
        excluded_lr = []
        append_excluded_lr = excluded_lr.append
        clear_excluded_lr = excluded_lr.clear
        for orpha, orpha_idx in self.dise_dict.items():
            # step 1, observed lr
            orpha_observed = get_orpha_observed(orpha)
            get_phe_observed_lr = orpha_observed.get
            for phe in positive_phe_list:
                phe_idx = get_phe_idx(phe)
                lr = get_phe_observed_lr(phe_idx, self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR)
                append_observed_lr(lr)
            # step 2, excluded lr
            orpha_excluded = get_orpha_excluded(orpha)
            get_phe_excluded_lr = orpha_excluded.get
            for phe in negative_phe_list:
                phe_idx = get_phe_idx(phe)
                record_lr = get_phe_excluded_lr(phe_idx, 0)
                append_excluded_lr(record_lr if record_lr > 0 else 1 / (1 - self.get_phe_background_frequency(phe)))

            composited_lr = np.prod(observed_lr) * np.prod(excluded_lr)
            posttest_odds = self.pretest_odds * composited_lr
            posttest_prob = posttest_odds / (1 + posttest_odds)
            disease_posterior_probability[orpha_idx] = posttest_prob

            clear_observed_lr()
            clear_excluded_lr()

        return disease_posterior_probability

    def get_phe_background_frequency(self, phe):
        idx = self.phe_dict.get(phe, -1)
        if idx == -1:
            return self.default_phe_background_freq
        else:
            return self.phe_background_freq[idx]

    def lr_sparse_matrix_to_dict(self, lr_sparse_mat):
        lr = {}
        for orpha, orpha_idx in self.dise_dict.items():
            orpha_lr_idx = lr_sparse_mat[orpha_idx].indices
            orpha_lr_data = lr_sparse_mat[orpha_idx].data
            lr[orpha] = dict(zip(orpha_lr_idx, orpha_lr_data))
        return lr

    def hpo_idx_transfer(self):
        orpha_hpo_idx = 0
        phe_hpo_idx = 0
        while orpha_hpo_idx < len(self.all_orpha_hpo):
            while True:
                if self.all_phe_hpo[phe_hpo_idx] == self.all_orpha_hpo[orpha_hpo_idx]:
                    self.phe_hpo_idx_to_orpha_hpo_idx[phe_hpo_idx] = orpha_hpo_idx
                    self.orpha_hpo_idx_to_phe_hpo_idx[orpha_hpo_idx] = phe_hpo_idx
                    break
                phe_hpo_idx += 1
            orpha_hpo_idx += 1