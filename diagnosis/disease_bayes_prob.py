import numpy as np


class DiseaseBayesProb:
    DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR = 0.01

    def __init__(self, dise_dict=None, phe_dict=None, phe_plus_dict=None, phe_background_freq=None, observed_lr=None, excluded_lr=None, dise_anno_phe=None):
        """
        Using this class to calculate the bayesian probability of ORPHA diseases
        :param dise_dict:
        :param phe_dict:
        :param phe_plus_dict:
        :param phe_background_freq:
        :param observed_lr:
        :param excluded_lr:
        """
        self.dise_dict = dise_dict
        self.phe_dict = phe_dict
        self.phe_plus_dict = phe_plus_dict
        self.dise_anno_phe = dise_anno_phe
        self.observed_lr = self.lr_sparse_matrix_to_dict(observed_lr)
        self.excluded_lr = self.lr_sparse_matrix_to_dict(excluded_lr)
        self.phe_background_freq = phe_background_freq
        self.default_phe_background_freq = 1 / len(self.phe_plus_dict)
        self.pretest_odds = 1 / len(self.dise_dict)

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
            # step 1
            orpha_observed = get_orpha_observed(orpha)
            get_phe_observed_lr = orpha_observed.get
            for phe in positive_phe_list:
                phe_idx = get_phe_idx(phe)
                append_observed_lr(get_phe_observed_lr(phe_idx, self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR))
            # step 2
            orpha_excluded = get_orpha_excluded(orpha)
            get_phe_excluded_lr = orpha_excluded.get
            for phe in negative_phe_list:
                phe_idx = get_phe_idx(phe)
                record_lr = get_phe_excluded_lr(phe_idx, 0)
                append_excluded_lr(record_lr if record_lr > 0 else 1 / (1 - self.get_phe_background_frequency(phe)))
            # step 3
            composited_lr = np.prod(observed_lr) * np.prod(excluded_lr)
            posttest_odds = self.pretest_odds * composited_lr
            posttest_prob = posttest_odds / (1 + posttest_odds)
            disease_posterior_probability[orpha_idx] = posttest_prob

            clear_observed_lr()
            clear_excluded_lr()

        return disease_posterior_probability

    def evaluate_disease_confusion_prob(self, dise):
        dise_phe_idx = [self.phe_plus_dict[list(self.phe_dict.keys())[i]] for i in self.dise_anno_phe[self.dise_dict[dise]].indices]
        dise_phe_freq = self.dise_anno_phe[self.dise_dict[dise]].data
        disease_posterior_probability = np.zeros(len(self.dise_dict))
        # hooks
        get_orpha_observed = self.observed_lr.get
        observed_lr = []
        append_observed_lr = observed_lr.append
        clear_observed_lr = observed_lr.clear
        for orpha, orpha_idx in self.dise_dict.items():
            # step 1
            orpha_observed = get_orpha_observed(orpha)
            get_phe_observed_lr = orpha_observed.get
            for i in range(len(dise_phe_idx)):
                phe_idx = dise_phe_idx[i]
                append_observed_lr(get_phe_observed_lr(phe_idx, self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR)*dise_phe_freq[i])
            # step 2
            composited_lr = np.prod(observed_lr)
            posttest_odds = self.pretest_odds * composited_lr
            posttest_prob = posttest_odds / (1 + posttest_odds)
            disease_posterior_probability[orpha_idx] = posttest_prob

            clear_observed_lr()

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
