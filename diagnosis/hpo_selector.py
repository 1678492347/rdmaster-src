import numpy as np
from scipy import sparse


class HpoSelector:
    def __init__(self, dise_dict=None, phe_dict=None, dise_phe_max_pro=None):
        self.y = np.array(list(dise_dict.values()))
        self.x = np.array(list(phe_dict.values()))
        self.dise_phe_max_pro = dise_phe_max_pro
        self.phe_dise_max_pro = sparse.csc_matrix(dise_phe_max_pro.toarray())
        self.top_n = 1
        self.class_weight = None
        self.positive_weight = None
        self.negative_weight = None

    def set_class_weight(self, class_weight):
        self.class_weight = class_weight
        # self.positive_weight = sparse.csr_matrix(self.dise_phe_max_pro.toarray().T * self.class_weight)
        # self.negative_weight = sparse.csr_matrix((1 - self.dise_phe_max_pro.toarray().T) * self.class_weight)

    def calculate_gini_gain_and_info_gain(self, get_gini_gain=False, get_info_gain=False):
        gini_gain = np.zeros(len(self.x)) if get_gini_gain else None
        info_gain = np.zeros(len(self.x)) if get_info_gain else None
        cur_gini, cur_entropy = self.gini_and_entropy(self.y, self.class_weight, get_gini_gain, get_info_gain)

        top_n_x = set()
        top_n_y = np.argsort(-self.class_weight)[:self.top_n]
        for i in top_n_y:
            top_n_x.update(set(self.dise_phe_max_pro[i].indices))

        for i in top_n_x:
            condition_gini, condition_entropy = self.con_gini_and_entropy(i, self.y, self.class_weight, get_gini_gain, get_info_gain)
            if get_gini_gain:
                gini_gain[i] = cur_gini - condition_gini
            if get_info_gain:
                info_gain[i] = cur_entropy - condition_entropy

        return cur_gini, gini_gain, info_gain

    def gini_and_entropy(self, y, weight=None, get_gini=False, get_entropy=False):
        if weight is None:
            weight = np.array([1.0] * len(y))

        weight_sum = np.sum(weight)

        # prob = np.array([np.sum(weight[y == i]) / weight_sum for i in np.unique(y)])
        prob = np.array([i / weight_sum for i in weight])
        gini = np.sum([prob_i * (1 - prob_i) for prob_i in prob]) if get_gini else None
        entropy = -np.sum([prob_i * np.log2(prob_i) for prob_i in prob]) if get_entropy else None
        return gini, entropy

    def con_gini_and_entropy(self, x, y, weight=None, get_gini=False, get_entropy=False):
        if weight is None:
            weight = np.array([1.0] * len(y))
        weight_sum = np.sum(weight)

        condition_gini = 0
        condition_entropy = 0

        full_y = np.zeros(len(self.y))
        full_y[self.phe_dise_max_pro[:, x].indices] = self.phe_dise_max_pro[:, x].data

        positive_index = np.where(full_y > 0)
        # positive_index = self.positive_weight[x].indices

        positive_y = y[positive_index]

        positive_weight = weight[positive_index] * full_y[positive_index]
        # positive_weight = self.positive_weight[x].data

        positive_prob = np.sum(positive_weight) / weight_sum

        positive_gini, positive_entropy = self.gini_and_entropy(positive_y, positive_weight, get_gini, get_entropy)
        if get_gini:
            condition_gini += positive_gini * positive_prob
        if get_entropy:
            condition_entropy += positive_entropy * positive_prob

        opposite_full_y = 1 - full_y

        negative_index = np.where(opposite_full_y > 0)
        # negative_index = self.negative_weight[x].indices

        negative_y = y[negative_index]

        negative_weight = weight[negative_index] * opposite_full_y[negative_index]
        # negative_weight = self.negative_weight[x].data

        negative_prob = np.sum(negative_weight) / weight_sum

        negative_gini, negative_entropy = self.gini_and_entropy(negative_y, negative_weight, get_gini, get_entropy)
        if get_gini:
            condition_gini += negative_gini * negative_prob
        if get_entropy:
            condition_entropy += negative_entropy * negative_prob

        return condition_gini, condition_entropy

    def gini_gain_and_info_gain(self, x, y, weight=None, get_gini_gain=False, get_info_gain=False):
        if weight is None:
            weight = np.array([1.0] * len(y))
        cur_gini, cur_entropy = self.gini_and_entropy(y, weight, get_gini_gain, get_info_gain)
        condition_gini, condition_entropy = self.con_gini_and_entropy(x, y, weight, get_gini_gain, get_info_gain)
        gini_gain = cur_gini - condition_gini if get_gini_gain else None
        info_gain = cur_entropy - condition_entropy if get_info_gain else None
        return gini_gain, info_gain
