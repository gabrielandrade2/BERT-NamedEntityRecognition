import re

import pandas as pd
import umls_api

from BERT.predict import predict
from util import iob_util
from BERT.util.bert_utils import normalize_dataset, load_model
from util.xml_parser import xml_to_articles, __preprocessing, drop_texts_with_mismatched_tags, \
    convert_xml_to_iob_list
from util.text_utils import split_sentences
from seqeval.metrics import accuracy_score, f1_score, precision_score, classification_report
from seqeval.scheme import IOB2
from dnorm_j import DNorm


def list_size(list):
    return sum([len(t) for t in list])

def flatten_list(list):
    flat_list = [item for sublist in list for item in sublist]
    return flat_list

if __name__ == '__main__':
    ##### Load model #####
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    model, tokenizer, id2label = load_model(MODEL, '../BERT/out')
    TAG_LIST = ['d']

    #### Load data #####
    # Get clean articles from file to tag
    xmlFile = '../data/drugHistoryCheck.xml'
    texts = xml_to_articles(xmlFile)
    texts = __preprocessing(texts)
    texts = split_sentences(texts)
    texts = drop_texts_with_mismatched_tags(texts)
    # Remove tags
    texts = [re.sub('<[^>]*>', '', t) for t in texts]

    # Get iob info from xml as ground true labels
    original_sentences, original_labels = convert_xml_to_iob_list(xmlFile, TAG_LIST, should_split_sentences=True)

    ##### Tokenize text for BERT #####
    # print(sum([len(t) for t in texts]))
    texts = [tokenizer.tokenize(t) for t in texts]
    data_x = [tokenizer.convert_tokens_to_ids(['[CLS]'] + t) for t in texts]

    # Normalize to same tokenization as BERT
    original_sentences, original_labels = normalize_dataset(original_sentences, original_labels, tokenizer)

    ##### Extract drug names #####
    tags = predict(model, data_x)
    labels = [[id2label[t] for t in tag] for tag in tags]
    data_x = [tokenizer.convert_ids_to_tokens(t)[1:] for t in data_x]

    # Remove pad tags from labels
    labels = [sent_label[:len(sent)] for sent, sent_label in zip(data_x, labels)]

    ##### Insanity check #####
    assert list_size(original_sentences) == list_size(data_x) == list_size(original_labels) == list_size(labels)

    ##### Save output iob file #####
    correct = 0
    f = open("out/iob_predict_" + xmlFile.split('/')[-1] + ".iob", 'w')
    for original_sentence, original_sentence_label, output_sentence, predict_sentence_label in zip(original_sentences, original_labels, data_x, labels):
        for original_char, original_char_label, output_char, predict_char_label in zip(original_sentence, original_sentence_label, output_sentence, predict_sentence_label):
            line = original_char + '\t' + original_char_label + '\t' + output_char + '\t' + predict_char_label + '\n'
            f.write(line)
            if original_char_label == predict_char_label:
                correct = correct + 1
        f.write('\n')
    f.close()

    ###### Calculate perfromance metrics #####
    print('Accuracy: ' + str(accuracy_score(original_labels, labels)))
    print('Precision: ' + str(precision_score(original_labels, labels)))
    print('F1 score: ' + str(f1_score(original_labels, labels)))
    #print(classification_report(original_labels, labels))
    print(classification_report(original_labels, labels, mode='strict', scheme=IOB2))

    output = pd.DataFrame()
    i = 0
    ##### Match tags to UMLS ####
    for sent_number in range(len(data_x)):

        print('Sentence', sent_number, ' of ', len(data_x))

        ne_dict = iob_util.convert_iob_to_dict(data_x[sent_number], labels[sent_number])

        # Normalize
        normalized_entities = list()
        normalization_model = DNorm.from_pretrained()
        for entry in ne_dict:
            named_entity = entry['word']
            normalized_named_entity = normalization_model.normalize(named_entity)
            normalized_entities.append(normalized_named_entity)
        df = pd.DataFrame(ne_dict)
        df['normalized'] = normalized_entities

        # Search on UMLS
        cuis = list()
        for entity in normalized_entities:
            results = umls_api.API(api_key='').term_search(entity)
            try:
                i = i + 1
                cui = results['result']['results'][0]['ui']
                print(cui)
            except Exception:
                cui = 0
            cuis.append(cui)
        df['cui'] = cuis
        df.insert(0, 'Sentence', sent_number)
        output = output.append(df, ignore_index=True)

        # Search on MedDRA


    # Output to csv
    output.to_csv("output.csv", sep=";")