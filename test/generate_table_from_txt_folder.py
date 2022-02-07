import glob

import pandas as pd

from BERT.predict import *
from util.bert import bert_utils, iob_util

if __name__ == '__main__':
    # Load BERT model
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    model, tokenizer, vocabulary = load_model(MODEL, '../BERT/out')

    # Get file list
    DIRECTORY = "../data/Croudworks薬歴/txt/"
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

        ne_dict = list()
        for sent, label in zip(sentences, labels):
            ne_dict.extend(iob_util.convert_iob_to_dict(sent, label[:len(sent)]))

        # Consolidate results in output variable
        drug = metadata[1]
        if drug in output_dict:
            drug_dict = output_dict[drug]
        else:
            drug_dict = {}

        for named_entity in ne_dict:
            word = named_entity['word']
            if word in drug_dict:
                count = drug_dict[word] + 1
            else:
                count = 1
            drug_dict[word] = count
        output_dict[drug] = drug_dict

    print(output_dict)
    output_table = pd.DataFrame.from_dict(output_dict, orient='index')
    output_table.fillna(0)