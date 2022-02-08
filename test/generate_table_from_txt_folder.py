import glob

import pandas as pd
from dnorm_j import DNorm

from BERT.predict import *
from util.bert import bert_utils, iob_util

def convert_to_dict(sentences, labels):
    ne_dict = list()
    for sent, label in zip(sentences, labels):
        ne_dict.extend(iob_util.convert_iob_to_dict(sent, label))
    return ne_dict

def normalize_entities(named_entities):
    normalized_entities = list()
    normalization_model = DNorm.from_pretrained()
    for entry in named_entities:
        entry['normalized_word'] = normalization_model.normalize(entry['word'])
        normalized_entities.append(entry)
    return normalized_entities

def consolidate_table_data(drug, output_dict, ne_dict):
    if drug in output_dict:
        drug_dict = output_dict[drug]
    else:
        drug_dict = {}

    for named_entity in ne_dict:
        word = named_entity['normalized_word']
        if word in drug_dict:
            count = drug_dict[word] + 1
        else:
            count = 1
        drug_dict[word] = count
    output_dict[drug] = drug_dict
    return output_dict

def table_post_process(table):
    # Order drugs by number of ADE events
    table['sum_col'] = table.sum(axis=1)
    table = table.sort_values('sum_col', ascending=False)
    table = table.drop(columns=["sum_col"])
    table = table.head(50)

    # Order ADE by numer of events
    table = table[table.sum(0).sort_values(ascending=False)[:50].index]

    return table

if __name__ == '__main__':
    # Load BERT model
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    model, tokenizer, vocabulary = load_model(MODEL, '../BERT/out')

    # Get file list
    DIRECTORY = "../data/Croudworks薬歴/txt-jp-drug/"
    file_list = glob.glob(DIRECTORY + '*.txt')

    output_dict = {}

    for i in range(len(file_list)):
        file = file_list[i]

        print('File', i, 'of', len(file_list))
        print(file)

        f = open(file, 'r')
        lines = f.readlines()

        # Get file metadata from first line
        metadata = lines[0].replace('%', '').split(',')
        # ['ID', 'Drug', 'Adverse Event', 'Place'])  # ID,  薬剤名, 有害事象, 想定した服薬指導実施場所

        # Apply the model to extract symptoms
        sentences_embeddings = bert_utils.prepare_sentences(lines[1:], tokenizer)
        tags = predict(model, sentences_embeddings)
        labels = convert_prediction_to_labels(tags, vocabulary)
        sentences = [tokenizer.convert_ids_to_tokens(t)[1:] for t in sentences_embeddings]
        labels = remove_label_padding(sentences, labels)
        ne_dict = convert_to_dict(sentences, labels)
        ne_dict = normalize_entities(ne_dict)

        # Consolidate results in output variable
        drug = metadata[1]
        output_dict = consolidate_table_data(drug, output_dict, ne_dict)

    output_table = pd.DataFrame.from_dict(output_dict, orient='index').fillna(0)
    output_table = table_post_process(output_table)
    output_table.to_excel('../data/output-jp.xlsx')
