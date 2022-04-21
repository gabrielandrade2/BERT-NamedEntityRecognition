import pandas as pd
from thefuzz import fuzz
from tqdm import tqdm

from knowledge_bases.meddra import MedDRADatabase
from util import iob_util
from util.Dataset import YakurekiTxtDataset


def match_llt(term, llt_list, database):
    temp = list()
    for index, llt in llt_list.iterrows():
        score = fuzz.token_set_ratio(term, llt['llt_kanji'])
        temp.append((score, llt))
    preferred_candidates = sorted(temp, key=lambda i: i[0], reverse=True)

    if preferred_candidates[0][0] > 0:
        output = list()
        for i in range(3):
            if i >= len(preferred_candidates):
                output.extend(['', '', ''])
            pc = preferred_candidates[i]
            score = pc[0]
            llt_code = pc[1]['llt_code']
            llt_term = pc[1]['llt_kanji']
            pt_term = database.get_pt_j_from_llt_code(llt_code)
            output.extend([llt_term, llt_code, pt_term[0], pt_term[1], score])
        return output
    return ["" for x in range(15)]


if __name__ == '__main__':
    dataset = YakurekiTxtDataset("/Users/gabriel-he/Documents/datasets/薬歴/薬歴_タグ付け済_中江")

    database = MedDRADatabase('/Users/gabriel-he/Documents/git/meddra-sqlite/db/meddra.sqlite3')
    database.open_connection()
    llt_list = database.get_all_llt_j()

    output_dict = {}
    i = 0
    for tagged_text in tqdm(dataset):
        # i += 1
        # if i == 10:
        #     break
        try:
            entities = list(
                map(lambda x: x[3], iob_util.convert_xml_to_taglist(tagged_text, ['d'], ignore_mismatch_tags=False)[1]))

            for entity in entities:
                if entity not in output_dict:
                    meddra = match_llt(entity, llt_list, database)
                    output_dict[entity] = [1] + meddra
                else:
                    temp = output_dict[entity]
                    output_dict[entity][0] += 1
        except Exception as e:
            print("Failed to process text")
            print(tagged_text)
            print(e)

    rows = list()
    for key in output_dict.keys():
        rows.append([key] + output_dict[key])
    df = pd.DataFrame(rows, columns=['entity', 'frequency', 'llt_term1', 'llt_code1', 'pt_term1', 'pt_code1', 'score1',
                                     'llt_term2', 'llt_code2', 'pt_term2', 'pt_code2', 'score2',
                                     'llt_term3', 'llt_code3', 'pt_term3', 'pt_code3', 'score3'])
    df.sort_values('frequency', ascending=False, inplace=True)
    df.to_excel('Yakureki_d_statistics_new.xlsx', index=False)
