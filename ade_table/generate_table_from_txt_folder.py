import glob

import pandas as pd
import torch
from dnorm_j import DNorm

from BERT.Model import NERModel
from BERT.predict import *
from util.ade_table_utils import *

if __name__ == '__main__':
    # Load BERT model
    model = NERModel.load_transformers_model('cl-tohoku/bert-base-japanese-char-v2', '../out/out_IM_v6')

    # Get file list
    DIRECTORY = "../data/Croudworks薬歴/txt-jp-drug/"
    file_list = glob.glob(DIRECTORY + '*.txt')

    output_dict = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(len(file_list)):
        file = file_list[i]

        print(file, '\n', 'File', i + 1, 'of', len(file_list), end='\r')

        f = open(file, 'r')
        lines = f.readlines()

        # Get file metadata from first line
        metadata = lines[0].replace('%', '').split(',')
        # ['ID', 'Drug', 'Adverse Event', 'Place'])  # ID,  薬剤名, 有害事象, 想定した服薬指導実施場所

        # Apply the model to extract symptoms
        tokenizer = model.tokenizer
        vocabulary = model.vocabulary

        sentences_embeddings = bert_utils.prepare_sentences(lines[1:], tokenizer)
        tags = model.predict(sentences_embeddings)
        labels = convert_prediction_to_labels(tags, vocabulary)
        sentences = [tokenizer.convert_ids_to_tokens(t)[1:] for t in sentences_embeddings]
        labels = remove_label_padding(sentences, labels)
        ne_dict = convert_labels_to_dict(sentences, labels)
        ne_dict = normalize_entities(ne_dict, DNorm.from_pretrained())

        # Consolidate results in output variable
        drug = metadata[1]
        output_dict = consolidate_table_data(drug, output_dict, ne_dict)

    output_table = pd.DataFrame.from_dict(output_dict, orient='index').fillna(0)
    output_table = table_post_process(output_table)
    output_table.to_excel('../data/output-from-txt-folder.xlsx')
