import pandas as pd
from diagnosis.hpo_disease import *


def load_disease_map(phenotype_hpoa_path, hpo_id_to_name):
    phenotype_hpoa = pd.read_csv(phenotype_hpoa_path, sep='\t', header=0, dtype=str, skiprows=4)
    phenotype_hpoa.rename(columns={'#DatabaseID': 'DatabaseID'}, inplace=True)
    orpha_phenotype_hpoa = phenotype_hpoa[phenotype_hpoa['DatabaseID'].str.contains('ORPHA')]           # only ORPHA
    orpha_phenotype_hpoa = orpha_phenotype_hpoa.loc[orpha_phenotype_hpoa['Aspect'].isin(['P'])]         # only PHENOTYPIC_ABNORMALITY
    orpha_phenotype_hpoa = orpha_phenotype_hpoa.loc[~orpha_phenotype_hpoa['Qualifier'].isin(['NOT'])]   # no Excluded
    orpha_phenotype_hpoa = orpha_phenotype_hpoa.loc[:, ['DatabaseID', 'DiseaseName', 'HPO_ID', 'Frequency']]

    all_orpha = list({}.fromkeys(orpha_phenotype_hpoa['DatabaseID'].values).keys())
    orpha_phenotype_hpoa.set_index('DatabaseID', inplace=True)

    disease_map = {}
    for orpha in all_orpha:
        if len(orpha_phenotype_hpoa.loc[orpha].shape) == 1:
            orpha_name = orpha_phenotype_hpoa.loc[orpha, 'DiseaseName']
            phenotypic_abnormalities = [{
                'hpo_id': orpha_phenotype_hpoa.loc[orpha][1],
                'hpo_name': hpo_id_to_name.get(orpha_phenotype_hpoa.loc[orpha][1]),
                'freq_id': orpha_phenotype_hpoa.loc[orpha][2],
                'freq_name': hpo_id_to_name.get(orpha_phenotype_hpoa.loc[orpha][2])
            }]
        else:
            orpha_name = orpha_phenotype_hpoa.loc[orpha, 'DiseaseName'].values[0]
            phenotypic_abnormalities = [{
                'hpo_id': anno[1],
                'hpo_name': hpo_id_to_name.get(anno[1]),
                'freq_id': anno[2],
                'freq_name': hpo_id_to_name.get(anno[2])
            } for anno in orpha_phenotype_hpoa.loc[orpha].values]
        disease_map[orpha] = HpoDisease(
            disease_id=orpha,
            name=orpha_name,
            annotations=phenotypic_abnormalities,
            prevalence=1
        )
    return all_orpha, disease_map


def induced_disease_graph_map(disease_map, hpo_onto_graph, up=True, per_reduce=5):
    induced_disease_map = {}
    for orpha, hpo_disease in disease_map.items():
        induced_disease_map[orpha] = InducedDiseaseGraph(
            hpo_disease=hpo_disease,
            hpo_onto_graph=hpo_onto_graph,
            up=up,
            per_reduce=per_reduce
        )
    return induced_disease_map

