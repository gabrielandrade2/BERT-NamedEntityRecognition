import re

import pandas as pd
import os
import glob
import mojimoji

from BERT.predict import load_model, predict
from BERT.util.iob_util import convert_iob_to_xml, convert_iob_to_dict
from BERT.util.xml_parser import xml_to_articles, __preprocessing

if __name__ == '__main__':
    # Load the model and file list
    directory = "../data/Croudworks薬歴/"
    file_list = glob.glob(directory + '*.xlsx')

    # load model
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    model, tokenizer, id2label = load_model(MODEL, 'out')

    # Get articles from file
    xmlFile = '../data/drugHistoryCheck.xml'
    texts = xml_to_articles(xmlFile)
    texts = __preprocessing(texts)

    # Remove tags
    texts = [re.sub('<[^>]*>', '', t) for t in texts]

    # Convert text to NFKC standard
    texts = [mojimoji.han_to_zen(t, kana=False) for t in texts]

    # Tokenize text for BERT
    texts = [tokenizer.tokenize(t) for t in texts]
    data_x = [tokenizer.convert_tokens_to_ids(['[CLS]'] + t) for t in texts]

    # Extract drug names
    tags = predict(model, data_x)
    labels = [[id2label[t] for t in tag] for tag in tags]
    data_x = [tokenizer.convert_ids_to_tokens(t)[1:] for t in data_x]

    flat_data_x = [item for sublist in data_x for item in sublist]
    flat_labels = [x[1][:len(x[0])] for x in zip(data_x, labels)]
    flat_labels = [item for sublist in flat_labels for item in sublist]

    r = open("../old/iob.iob", 'r')
    f = open("iob.iob", 'w')
    lines = r.readlines()

    i = 0
    for line in lines:
        line = line.replace('\n', '')
        line = line + '\t' + flat_labels[i] + '\t' + flat_data_x[i] + '\n'
        f.write(line)
        i = i + 1
