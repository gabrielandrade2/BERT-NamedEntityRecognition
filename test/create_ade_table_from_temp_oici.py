import glob
import os.path as path
import re

import pandas as pd
from tqdm import tqdm

from ade_table import ade_table
from knowledge_bases.OICI_database import OICIDatabase, OICINormalizer


def wrapper(data):
    normalizer = data[0]
    l = data[1]
    return [normalizer.normalize(t) for t in l]


if __name__ == '__main__':
    files = sorted(glob.glob('/Users/gabriel-he/Documents/OICI/oici0914/Extracted Data/*_out_IM_v6_negative.xlsx'))
    # normalizer = ManbyoNormalizer(ManbyoDict('data/DIC_C_MANBYO_作業_20220331_v1.0.csv'), threshold=70)
    normalizer = OICINormalizer(OICIDatabase())

    for i, file in enumerate(files):
        all_drugs = []
        all_symptoms = []
        print(file)
        df = pd.read_excel(file)
        df = df.drop(df[df['CN'] == '[\'failed\']'].index)

        drugs = [re.sub(r'[\[\]\']', '', item) for item in df['レジメン名'].to_list()]
        drugs = [[item] if item != 'nan' else [] for item in drugs]
        symptoms = [re.sub(r'[\[\]\']', '', item).split(',') for item in df['CN'].to_list()]
        # normalized_symptoms = symptoms

        # from multiprocessing import Pool
        # with Pool() as pool:
        #     normalized_symptoms = list(tqdm(pool.imap(wrapper, zip(itertools.repeat(normalizer), symptoms)), total=len(symptoms)))
        #     pool.close()
        #     pool.join()

        #
        normalized_symptoms = [[normalizer.normalize(t) for t in l] for l in tqdm(symptoms)]
        #
        # pd.DataFrame([symptoms, normalized_symptoms]).transpose().to_excel(file.split('/')[-1]+'_oici_human_check.xlsx')

        drug_norm = pd.read_excel('/Users/gabriel-he/Documents/OICI/regiment drug.xlsx')
        # drug_norm['Re'] = [unicodedata.normalize('NFKC', item) for item in drug_norm['Re']]

        drugs = [[drug_norm[drug_norm['Re'] == d]['Anticancer drug'].item() if drug_norm[drug_norm['Re'] == d][
            'Re'].any() else d for d in drug_list] for drug_list in drugs]

        normalized_symptoms = [list(filter(bool, l)) for l in tqdm(normalized_symptoms)]
        print('empty:', len([l for l in normalized_symptoms if len(l) == 0]))

        all_drugs.extend(drugs)
        all_symptoms.extend(normalized_symptoms)
        # table = ade_table.from_lists(drugs, normalized_symptoms, remove_duplicates=True)
        # table.filter(num_symptoms=40)
        # table.to_excel(file.replace('.xlsx', '_ade_table_no_duplicates.xlsx'))
        table = ade_table.from_lists(all_drugs, all_symptoms, remove_duplicates=True)
        dir = path.dirname(file)
        table.to_excel(path.join(dir, path.basename(file).replace('.xlsx', '_ade_table_negative.xlsx')))
