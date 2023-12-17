import math
import networkx


class HpoDisease:
    def __init__(self, name=None, name_cn=None, disease_id=None, annotations=None,
                 modes_of_inheritance=None, prevalence=None, onset_age=None, death_age=None):
        self.name = name
        self.name_cn = name_cn
        self.disease_id = disease_id
        self.annotations = annotations
        self.modes_of_inheritance = modes_of_inheritance
        self.prevalence = prevalence
        self.onset_age = onset_age
        self.death_age = death_age


class InducedDiseaseGraph:
    ROOT_HPO = 'HP:0000001'
    DIFF_FREQ_SCORE = {
        'Obligate': 1,
        'Very frequent': 0.9,
        'Frequent': 0.55,
        'Occasional': 0.17,
        'Very rare': 0.03,
        'Excluded': 0
    }
    """
    Create the induced graph of the HPO terms used to annotate the disease.
    We weight the frequency downwards according to the number of links (path length).
    That is, if the path length from a direct annotation to an ancestor is k,
    then we multiple the frequency of the annotation by (1/math.pow(per_reduce, k)).
    The children HPO is not considered yet.
    """
    def __init__(self, hpo_disease=None, hpo_onto_graph=None, up=False, per_reduce=1):
        self.hpo_disease = hpo_disease
        self.hpo2freq_score = {}
        self.hpo2anno_dis = {}

        for anno in hpo_disease.annotations:
            freq_score = self.DIFF_FREQ_SCORE.get(anno.get('freq_name'))
            hpo_id = anno.get('hpo_id')
            self.hpo2freq_score[hpo_id] = freq_score
            self.hpo2anno_dis[hpo_id] = 0
            if not up: continue
            distance = 1
            while True:
                hpo_terms = networkx.descendants_at_distance(hpo_onto_graph, hpo_id, distance)
                if len(hpo_terms) == 0: break
                for hpo in hpo_terms:
                    if hpo == self.ROOT_HPO:
                        continue
                    pre_freq_score = self.hpo2freq_score.get(hpo, 0)
                    cur_freq_score = freq_score / math.pow(per_reduce, distance)
                    if cur_freq_score > pre_freq_score:
                        self.hpo2freq_score[hpo] = cur_freq_score
                    pre_dis = self.hpo2anno_dis.get(hpo, 1000)
                    if distance < pre_dis:
                        self.hpo2anno_dis[hpo] = distance
                distance += 1
