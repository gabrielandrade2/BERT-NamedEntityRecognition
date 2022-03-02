import json
import os

import torch
from sklearn.model_selection import train_test_split
from transformers import BertForTokenClassification, BertJapaneseTokenizer

from BERT import bert_utils
from BERT.Model import NERModel
from util.xml_parser import convert_xml_file_to_iob_list


def train_from_xml_file(xmlFile, model_name, tag_list, output_dir):
    ##### Load the data #####
    sentences, tags = convert_xml_file_to_iob_list(xmlFile, tag_list, should_split_sentences=True)
    return train_from_sentences_tags_list(sentences, tags, model_name, output_dir)


def train_from_sentences_tags_list(sentences, tags, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ' + device)

    ##### Process dataset for BERT #####
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

    # Create vocabulary
    label_vocab = bert_utils.create_label_vocab(tags)
    with open(output_dir + '/label_vocab.json', 'w') as f:
        json.dump(label_vocab, f, ensure_ascii=False)

    ##### Split in train/validation #####
    train_x, validation_x, train_y, validation_y = train_test_split(sentences, tags, test_size=0.2)

    # Convert to BERT data model
    train_x, train_y = bert_utils.dataset_to_bert_input(train_x, train_y, tokenizer, label_vocab)
    validation_x, validation_y = bert_utils.dataset_to_bert_input(validation_x, validation_y, tokenizer, label_vocab)

    # Get pre-trained model and fine-tune it
    pre_trained_model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_vocab))
    model = NERModel(pre_trained_model, tokenizer, label_vocab, device=device)
    model.train(train_x, train_y, val=[validation_x, validation_y], outputdir=output_dir)

    return model
