import json
import os

from sklearn.model_selection import train_test_split
from transformers import BertForTokenClassification, BertJapaneseTokenizer
from BERT.train import train
from util.bert import bert_utils
from util.bert.xml_parser import convert_xml_to_iob_list

if __name__ == '__main__':

    xmlFile = '../data/drugHistoryCheck.xml'
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    TAG_LIST = ['d']
    output_dir = 'out'

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print("folder exists")

    ##### Load the data #####
    sentences, tags = convert_xml_to_iob_list(xmlFile, TAG_LIST, should_split_sentences=True)

    ##### Process dataset for BERT #####
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL)

    # Create vocabulary
    label_vocab = bert_utils.create_label_vocab(tags)
    with open(output_dir + '/label_vocab.json', 'w') as f:
        json.dump(label_vocab, f, ensure_ascii=False)

    ##### Split in train/test #####
    x_train, x_test, y_train, y_test = train_test_split(sentences, tags, test_size=0.2)

    # Convert to BERT data model
    input_x, input_y = bert_utils.dataset_to_bert_input(x_train, y_train, tokenizer, label_vocab)
    val_x, val_y = bert_utils.dataset_to_bert_input(x_test, y_test, tokenizer, label_vocab)

    # Get pre-trained model and fine-tune it
    model = BertForTokenClassification.from_pretrained(MODEL, num_labels=len(label_vocab))
    model = train(model, input_x, input_y, val=[val_x, val_y], outputdir=output_dir)

    #print(model)
    print()