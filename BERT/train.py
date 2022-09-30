import json
import os

import torch
from sklearn.model_selection import train_test_split
from transformers import BertForTokenClassification, BertJapaneseTokenizer

from BERT import bert_utils
from BERT.Model import NERModel
from util.iob_util import convert_xml_text_list_to_iob_list
from util.text_utils import split_sentences, exclude_long_sentences
from util.xml_parser import convert_xml_file_to_iob_list


def train_from_xml_file(xmlFile, model_name, tag_list, output_dir, parameters=None, attr_list=None,
                        should_split_sentences=True, device=None):
    ##### Load the data #####
    sentences, tags = convert_xml_file_to_iob_list(xmlFile, tag_list, attr_list=attr_list,
                                                   should_split_sentences=should_split_sentences)
    return train_from_sentences_tags_list(sentences, tags, model_name, output_dir, parameters, device)


def train_from_xml_texts(texts, model_name, tag_list, output_dir, parameters=None, attr_list=None,
                         should_split_sentences=True, device=None):
    if should_split_sentences:
        texts = split_sentences(texts)
    sentences, tags = convert_xml_text_list_to_iob_list(texts, tag_list, attr=attr_list)
    return train_from_sentences_tags_list(sentences, tags, model_name, output_dir, parameters, device)


def train_from_sentences_tags_list(sentences, tags, model_name, output_dir, parameters=None, device=None):
    os.makedirs(output_dir, exist_ok=True)

    sentences, tags = exclude_long_sentences(512, sentences, tags)

    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print('device: ' + device)

    ##### Process dataset for BERT #####
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

    # Create vocabulary
    label_vocab = bert_utils.create_label_vocab(tags)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + '/label_vocab.json', 'w') as f:
        json.dump(label_vocab, f, ensure_ascii=False)

    ##### Split in train/validation #####
    train_x, validation_x, train_y, validation_y = train_test_split(sentences, tags, test_size=0.1)

    # Convert to BERT data model
    train_x, train_y = bert_utils.dataset_to_bert_input(train_x, train_y, tokenizer, label_vocab)
    validation_x, validation_y = bert_utils.dataset_to_bert_input(validation_x, validation_y, tokenizer, label_vocab)

    # Get pre-trained model and fine-tune it
    pre_trained_model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_vocab))
    model = NERModel(pre_trained_model, tokenizer, label_vocab, device=device)
    model.train(train_x, train_y, parameters, val=[validation_x, validation_y], outputdir=output_dir)

    return model


def finetune_from_xml_file(xmlFile, model: NERModel, tag_list, output_dir, parameters=None, attr_list=None,
                           should_split_sentences=True):
    ##### Load the data #####
    sentences, tags = convert_xml_file_to_iob_list(xmlFile, tag_list, attr_list=attr_list,
                                                   should_split_sentences=should_split_sentences)
    return finetune_from_sentences_tags_list(sentences, tags, model, output_dir, parameters)


def finetune_from_xml_texts(texts, model: NERModel, tag_list, output_dir, parameters=None, attr_list=None,
                            should_split_sentences=True):
    if should_split_sentences:
        texts = split_sentences(texts)
    sentences, tags = convert_xml_text_list_to_iob_list(texts, tag_list, attr=attr_list)
    return train_from_sentences_tags_list(sentences, tags, model, output_dir, parameters)


def finetune_from_sentences_tags_list(sentences, tags, model: NERModel, output_dir=None, parameters=None,
                                      validation_ratio=0.1):
    sentences, tags = exclude_long_sentences(512, sentences, tags)

    ##### Split in train/validation #####
    train_x, validation_x, train_y, validation_y = train_test_split(sentences, tags, test_size=validation_ratio)

    # Convert to BERT data model
    train_x, train_y = bert_utils.dataset_to_bert_input(train_x, train_y, model.tokenizer, model.vocabulary)
    validation_x, validation_y = bert_utils.dataset_to_bert_input(validation_x, validation_y, model.tokenizer,
                                                                  model.vocabulary)

    # FineTune model
    if output_dir is None:
        output_dir = model.output_dir
    model.train(train_x, train_y, parameters, val=[validation_x, validation_y], outputdir=output_dir)

    with open(output_dir + '/label_vocab.json', 'w') as f:
        json.dump(model.vocabulary, f, ensure_ascii=False)

    return model
