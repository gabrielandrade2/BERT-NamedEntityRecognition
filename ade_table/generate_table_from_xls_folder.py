import glob
import re
import pandas as pd

from BERT.predict import *
from util.ade_table_utils import *
from BERT.util import bert_utils


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
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    model, tokenizer, vocabulary = load_model(MODEL, '../BERT/out')

    # Get file list
    DIRECTORY = "../data/Croudworks薬歴/"
    file_list = glob.glob(DIRECTORY + '[!~]*.xlsx')

    output_dict = {}
    full_ne_dict = list()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            sentences_embeddings = bert_utils.prepare_sentences(text.split('\n'), tokenizer)
            tags = predict(model, sentences_embeddings, device=device)
            labels = convert_prediction_to_labels(tags, vocabulary)
            sentences = [tokenizer.convert_ids_to_tokens(t)[1:] for t in sentences_embeddings]
            labels = remove_label_padding(sentences, labels)
            ne_dict = convert_labels_to_dict(sentences, labels)
            ne_dict = normalize_entities(ne_dict)
            full_ne_dict.append(ne_dict)

            # Consolidate results in output variable
            output_dict = consolidate_table_data(drug, output_dict, ne_dict)
        print('')

    output_table = pd.DataFrame.from_dict(output_dict, orient='index').fillna(0)
    output_table = table_post_process(output_table)
    output_table.to_excel('../data/output-from-xls-folder.xlsx')
