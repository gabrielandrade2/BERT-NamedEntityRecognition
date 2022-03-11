import glob
import re

import torch

from BERT.Model import NERModel
from BERT.predict import *
from knowledge_bases.meddra import *
from util.ade_table_utils import *


def get_drug(drugs, rownum):
    # In case it is empty, iterate back until we find the last drug listed
    i = 0
    drug = ""
    while not drug or drug == 'nan':
        # get drug name
        drug = str(drugs[rownum - i])
        #drug = re.search(r'[一-龯ぁ-ゔゞァ-・ヽヾ゛゜ーA-zＡ-ｚ0-9０-９]*', drug).group()
        i = i + 1
    return drug

if __name__ == '__main__':
    # Load BERT model
    model = NERModel.load_transformers_model('cl-tohoku/bert-base-japanese-char-v2', '../out/out_IM_v6')

    # Get file list
    DIRECTORY = "../data/Croudworks薬歴/"
    file_list = glob.glob(DIRECTORY + '[!~]*.xlsx')

    output_dict = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    database = MedDRADatabase('/Users/gabriel-he/Documents/git/meddra-sqlite/db/meddra.sqlite3')
    database.open_connection()
    normalization_model = MedDRAPatientFriendlyPTEntityNormalizer(
        database,
        MedDRAPatientFriendlyList('/Users/gabriel-he/Documents/MedDRA/patient-friendly_term_list_v24.1_J.xlsx')
    )

    for i in range(len(file_list)):
        file = file_list[i]

        print('\nFile', i + 1, 'of', len(file_list))
        print(file)

        xls = pd.ExcelFile(file)
        sheetX = xls.parse(0)

        # Get relevant columns
        try:
            texts = sheetX['患者像と薬歴（SOAPのS）']
            drugs = sheetX['薬剤名']
            # ades = sheetX['想定した有害事象']
            # locations = sheetX['想定した服薬指導実施場所（調剤薬局，病院（外来），病院（病棟））']
        except KeyError:
            print("Sheet not found, Skipping file")
            continue

        # Skip the first item as it is the 例 line
        for text_num in range(1, len(texts)):
            print('Text', text_num + 1, 'of', len(texts), end='\r')

            text = texts[text_num]
            # Skip empty texts
            if text != text:
                continue

            # Add \n after "。" which do not already have it
            text = re.sub('。(?=[^\n])', "。\n", text)

            drug = get_drug(drugs, text_num)

            # Apply the model to extract symptoms
            sentences = text.split('\n')
            sentences, labels = predict_from_sentences_list(model, sentences)
            ne_dict = convert_labels_to_dict(sentences, labels)
            ne_dict = normalize_entities(ne_dict, normalization_model)

            # Consolidate results in output variable
            output_dict = consolidate_table_data(drug, output_dict, ne_dict)
        print('')

    output_table = pd.DataFrame.from_dict(output_dict, orient='index').fillna(0)
    output_table = table_post_process(output_table)
    output_table.to_excel('../data/output-from-xls-folder_IM_v6-meddra-full.xlsx')

    generate_heatmap(output_table)
