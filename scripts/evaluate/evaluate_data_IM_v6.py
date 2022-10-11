import re

import pandas as pd
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, classification_report
from seqeval.scheme import IOB2

from BERT.Model import NERModel
from util.iob_util import convert_xml_text_list_to_iob_list
from util.text_utils import split_sentences, preprocessing


def list_size(list):
    return sum([len(t) for t in list])


def flatten_list(list):
    flat_list = [item for sublist in list for item in sublist]
    return flat_list


if __name__ == '__main__':
    ##### Load model #####
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = NERModel.load_transformers_model('cl-tohoku/bert-base-japanese-char-v2', '../../out/out_IM_v6_negative',
                                             device=device)
    tag_list = ['C']
    attr_list = ['MOD']

    #### Load data #####
    # Get clean articles from file to tag
    file = '../../data/DATA_IM_v6.txt'
    data = pd.read_csv(file, sep="	")
    texts_tagged = data['text_tagged'].tolist()
    texts_raw = data['text_raw'].tolist()

    # Preprocess
    texts_split = split_sentences(texts_tagged, True)
    texts_split = [t for t in texts_split if len(t) < 512]
    texts = preprocessing(texts_split, True)

    # Get iob info from xml as ground true labels
    original_sentences, original_labels, dropped = convert_xml_text_list_to_iob_list(texts, tag_list, attr_list,
                                                                                     ignore_mismatch_tags=True,
                                                                                     print_failed_sentences=True)
    # Remove tags
    texts = [re.sub('<[^>]*>', '', t) for t in texts]

    # Drop failed texts if any
    texts = [i for j, i in enumerate(texts) if j not in dropped]

    ##### Tokenize text for BERT #####
    data_x = model.prepare_sentences(texts)

    # Normalize to same tokenization as BERT
    original_sentences, original_labels = model.normalize_tagged_dataset(original_sentences, original_labels)

    ##### Sanity check #####
    print('Original sentences: ', list_size(original_sentences))
    print('Original labels: ', list_size(original_labels))
    print('Untagged sentences: ', list_size(texts))
    print('Predicted sentences: ', list_size([d[1:] for d in data_x]))

    to_remove = []
    for i, (s, l, d) in enumerate(zip(original_sentences, original_labels, data_x)):
        if len(s) != len(d[1:]) or len(l) != len(d[1:]):
            to_remove.append(i)

    original_sentences = [i for j, i in enumerate(original_sentences) if j not in to_remove]
    original_labels = [i for j, i in enumerate(original_labels) if j not in to_remove]
    data_x = [i for j, i in enumerate(data_x) if j not in to_remove]

    print('Original sentences: ', list_size(original_sentences))
    print('Original labels: ', list_size(original_labels))
    print('Untagged sentences: ', list_size(texts))
    print('Predicted sentences: ', list_size([d[1:] for d in data_x]))
    assert list_size(original_sentences) == list_size([d[1:] for d in data_x]) == list_size(original_labels)

    ##### Extract symptom names #####
    labels = model.predict(data_x)
    data_x = model.convert_ids_to_tokens(data_x)

    # Remove pad tags from labels
    labels = [sent_label[:len(sent)] for sent, sent_label in zip(data_x, labels)]

    ##### Sanity check #####
    print('Original sentences: ', list_size(original_sentences))
    print('Original labels: ', list_size(original_labels))
    print('Predicted sentences: ', list_size(data_x))
    print('Predicted labels: ', list_size(labels))
    assert list_size(original_sentences) == list_size(data_x) == list_size(original_labels) == list_size(labels)

    ##### Save output iob file #####
    correct = 0
    f = open("../../out/iob_predict_" + file.split('/')[-1] + "negative.iob", 'w')
    for original_sentence, original_sentence_label, output_sentence, predict_sentence_label in zip(original_sentences,
                                                                                                   original_labels,
                                                                                                   data_x, labels):
        for original_char, original_char_label, output_char, predict_char_label in zip(original_sentence,
                                                                                       original_sentence_label,
                                                                                       output_sentence,
                                                                                       predict_sentence_label):
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
    # print(classification_report(original_labels, labels))
    print(classification_report(original_labels, labels, mode='strict', scheme=IOB2))
