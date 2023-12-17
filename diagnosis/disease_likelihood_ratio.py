import numpy as np
from queue import Queue


class DiseaseLikelihoodRatio:
    DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR = 0.01
    DEFAULT_FALSE_NEGATIVE_OBSERVATION_OF_PHENOTYPE_LR = 0.01
    PHENOTYPIC_ABNORMALITY = 'HP:0000118'
    PARENT = 1
    
    def __init__(self, all_orpha_hpo=None, all_phe_hpo=None, all_orpha=None,  hpo_anc_mat=None, hpo_desc_mat=None,
                 hpo_background_freq=None, orpha2hpo=None, orpha_hpo_pro=None, orpha_hpo_max_pro=None):
        """
        Using this class to calculate the likelihood ratio is too time-costing,
        we have precomputed and stored the likelihood ratios between all phenotypes and diseases
        :param all_orpha_hpo:       all orpha related PHENOTYPIC_ABNORMALITY hpo
        :param all_phe_hpo:         all PHENOTYPIC_ABNORMALITY hpo
        :param all_orpha:           all ORPHA disease
        :param hpo_anc_mat:         hpo ancestor sparse matrix
        :param hpo_desc_mat:        hpo descendants sparse matrix
        :param hpo_background_freq  hpo background frequency calculated from disease hpo relation
        :param orpha2hpo:           disease annotated hpo relation sparse matrix
        :param orpha_hpo_pro:       disease related hpo probability sparse matrix
        :param orpha_hpo_max_pro:   disease related hpo max probability sparse matrix
        """
        self.all_orpha_hpo = all_orpha_hpo
        self.all_phe_hpo = all_phe_hpo
        self.all_phe_hpo_arr = np.array(list(all_phe_hpo.keys()))
        self.all_orpha = all_orpha
        self.hpo_anc_mat = hpo_anc_mat
        self.hpo_desc_mat = hpo_desc_mat
        self.hpo_background_freq = dict(zip(all_orpha_hpo, hpo_background_freq))
        self.orpha2hpo = orpha2hpo
        self.orpha_hpo_pro = orpha_hpo_pro
        self.orpha_hpo_max_pro = orpha_hpo_max_pro

        self.default_hpo_background_freq = 1/len(self.all_phe_hpo)

        self.phe_hpo_idx_to_orpha_hpo_idx = np.full(len(self.all_phe_hpo), -1, dtype=int)
        self.orpha_hpo_idx_to_phe_hpo_idx = np.full(len(self.all_orpha_hpo), -1, dtype=int)
        self.hpo_idx_transfer()

    def evaluate_disease_by_phenotype(self, phe_list, negated_phe_list):
        """
        :param phe_list:            observed TRUE phenotypes
        :param negated_phe_list:    excluded FALSE phenotypes
        :return: all ORPHA disease posterior probability
        """
        disease_prior_probability = 1/len(self.all_orpha)
        pretest_odds = disease_prior_probability / (1 - disease_prior_probability)

        disease_posterior_probability = np.zeros(len(self.all_orpha))
        for orpha_idx in range(len(self.all_orpha)):
            observed_lr = self.get_likelihood_ratios_for_observed_phenotype(phe_list, orpha_idx)
            excluded_lr = self.get_likelihood_ratios_for_excluded_phenotype(negated_phe_list, orpha_idx)
            composited_lr = np.prod(observed_lr) * np.prod(excluded_lr)
            posttest_odds = pretest_odds * composited_lr
            posttest_prob = posttest_odds / (1 + posttest_odds)
            disease_posterior_probability[orpha_idx] = posttest_prob
        return disease_posterior_probability

    def get_likelihood_ratios_for_observed_phenotype(self, phe_list, orpha_idx):
        lr_list = []
        orpha_anno_idx = self.orpha2hpo[orpha_idx].indices
        orpha_anno_phe_idx = self.orpha_hpo_idx_to_phe_hpo_idx[orpha_anno_idx]      # anno hpo index of all_phe_hpo
        orpha_rela_idx = self.orpha_hpo_max_pro[orpha_idx].indices
        orpha_rela_phe_idx = self.orpha_hpo_idx_to_phe_hpo_idx[orpha_rela_idx]

        for hpo in phe_list:
            hpo_background_freq = self.get_hpo_background_freq(hpo)

            # if hpo is directly annotated to this ORPHA disease
            if self.all_orpha_hpo[hpo] in orpha_anno_idx:
                numerator = self.orpha2hpo[orpha_idx].data[np.where(orpha_anno_idx == self.all_orpha_hpo[hpo])].item()
                denominator = hpo_background_freq
                lr_list.append(numerator/denominator)
                continue

            # if hpo is super_class of directly annotated phenotypes
            if self.all_orpha_hpo[hpo] in orpha_rela_idx:
                max_freq_of_desc = self.orpha_hpo_max_pro[orpha_idx].data[np.where(orpha_rela_idx == self.all_orpha_hpo[hpo])].item()
                lr_list.append(max_freq_of_desc/hpo_background_freq)
                continue

            # if hpo is sub_class of directly annotated phenotypes
            # TODO: re_design, cur_lr = anc_lr*(cur_bf/child_background_freq_sum)
            hpo_anc_idx = self.hpo_anc_mat[self.all_phe_hpo[hpo]].indices               # current hpo anc index of all_phe_hpo, not including itself
            hpo_anc_in_anno = np.intersect1d(hpo_anc_idx, orpha_anno_phe_idx)           # ann hpo anc index of all_phe_hpo
            if len(hpo_anc_in_anno) > 0:
                q = Queue()     # queue for BFS
                for anc_idx in hpo_anc_in_anno:
                    q.put([anc_idx, self.orpha2hpo[orpha_idx].data[np.where(orpha_anno_idx == self.phe_hpo_idx_to_orpha_hpo_idx[anc_idx])].item()])
                max_lr = self.bfs_to_get_hpo_induced_lr(q, hpo, hpo_anc_idx, hpo_background_freq)
                lr_list.append(max_lr)
                continue

            # if hpo has common ancestor with annotated phenotypes, not only PHENOTYPIC_ABNORMALITY
            # TODO: re_design, cur_lr = anc_lr*(cur_bf/child_background_freq_sum)
            hpo_anc_in_rela = np.intersect1d(hpo_anc_idx, orpha_rela_phe_idx)
            if len(hpo_anc_in_rela) != 1:
                q = Queue()  # queue for BFS
                for anc_idx in hpo_anc_in_rela:
                    if anc_idx != self.all_phe_hpo[self.PHENOTYPIC_ABNORMALITY]:
                        q.put([anc_idx, self.orpha_hpo_pro[orpha_idx].data[np.where(orpha_rela_idx == self.phe_hpo_idx_to_orpha_hpo_idx[anc_idx])].item()])
                max_lr = self.bfs_to_get_hpo_induced_lr(q, hpo, hpo_anc_idx, hpo_background_freq)
                lr_list.append(max_lr)
                continue

            # if hpo has common ancestor with annotated phenotypes, only PHENOTYPIC_ABNORMALITY
            if len(hpo_anc_in_rela) == 1 and hpo_anc_in_rela[0].item() == self.PHENOTYPIC_ABNORMALITY:
                lr_list.append(self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR)
            lr_list.append(self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR)

        return lr_list

    def get_likelihood_ratios_for_excluded_phenotype(self, negated_phe_list, orpha_idx):
        lr_list = []
        orpha_rela_idx = self.orpha_hpo_max_pro[orpha_idx].indices
        for hpo in negated_phe_list:
            hpo_background_freq = self.get_hpo_background_freq(hpo)

            # if hpo is not related to this ORPHA disease(annotated phenotype or super_class), lr should be higher(positive)
            if self.all_orpha_hpo[hpo] not in orpha_rela_idx:
                lr_list.append(1/(1-hpo_background_freq))

            # if hpo is related to this ORPHA disease, lr should be negative
            else:
                max_freq_of_desc = self.orpha_hpo_max_pro[orpha_idx].data[np.where(orpha_rela_idx == self.all_orpha_hpo[hpo])]
                lr_list.append(max((1-max_freq_of_desc)/(1-hpo_background_freq), self.DEFAULT_FALSE_NEGATIVE_OBSERVATION_OF_PHENOTYPE_LR))

        return lr_list

    def bfs_to_get_hpo_induced_lr(self, q, hpo, hpo_anc_idx, hpo_background_freq):
        max_lr = self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR
        min_inherited_freq = hpo_background_freq*self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_LR
        while q.qsize() > 0:
            size = q.qsize()
            for i in range(size):
                anc = q.get()
                child_idx = self.hpo_desc_mat[:, anc[0]].indices[
                    np.where(self.hpo_desc_mat[:, anc[0]].data == self.PARENT)]
                child_background_freq_sum = np.sum(np.vectorize(self.get_hpo_background_freq)(self.all_phe_hpo_arr[child_idx]))
                if self.all_phe_hpo[hpo] in child_idx:
                    hpo_inherited_freq = anc[1] * (hpo_background_freq / child_background_freq_sum)
                    max_lr = max(max_lr, hpo_inherited_freq / hpo_background_freq)

                cur_hpo_anc = np.intersect1d(child_idx, hpo_anc_idx)
                for anc_idx in cur_hpo_anc:
                    # inherit frequency from parent
                    inherited_freq = anc[1] * (self.get_hpo_background_freq(self.all_phe_hpo_arr[anc_idx]) / child_background_freq_sum)
                    # if inherited_freq is already very small, so hpo has little or no association with disease, skip
                    if inherited_freq < min_inherited_freq:
                        continue
                    else:
                        q.put([anc_idx, inherited_freq])
        return max_lr

    def get_hpo_background_freq(self, hpo):
        return self.hpo_background_freq.get(hpo, self.default_hpo_background_freq)

    def hpo_idx_transfer(self):
        all_orpha_hpo = list(self.all_orpha_hpo.keys())
        all_phe_hpo = list(self.all_phe_hpo.keys())
        orpha_hpo_idx = 0
        phe_hpo_idx = 0
        while orpha_hpo_idx < len(all_orpha_hpo):
            while True:
                if all_phe_hpo[phe_hpo_idx] == all_orpha_hpo[orpha_hpo_idx]:
                    self.phe_hpo_idx_to_orpha_hpo_idx[phe_hpo_idx] = orpha_hpo_idx
                    self.orpha_hpo_idx_to_phe_hpo_idx[orpha_hpo_idx] = phe_hpo_idx
                    break
                phe_hpo_idx += 1
            orpha_hpo_idx += 1
