import json
import os

from BERT.util.xml_parser import convert_xml_to_dataframe, convert_xml_to_iob_list
from BERT.train import train
from transformers import BertForTokenClassification, BertJapaneseTokenizer
from BERT.util import data_utils
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    xmlFile = '../data/drugHistoryCheck.xml'
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    output_dir = 'out'

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print("folder exists")

    ##### Load the data #####
    df = convert_xml_to_dataframe(xmlFile, ['m-key'])
    sentences, tags = convert_xml_to_iob_list(xmlFile, ['d'], False, True)

    ##### Split in train/test #####
    x_train, x_test, y_train, y_test = train_test_split(sentences, tags, test_size=0.2)

    ##### Process dataset for BERT #####
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL)

    # Create vocabulary
    label_vocab = data_utils.create_label_vocab(tags)
    with open(output_dir + '/label_vocab.json', 'w') as f:
        json.dump(label_vocab, f, ensure_ascii=False)

    # Add BERT tags
    input_x = [tokenizer.convert_tokens_to_ids(['[CLS]'] + x) for x in x_train]
    input_y = [data_utils.sent2input(['[PAD]'] + x, label_vocab) for x in y_train]
    val_x = [tokenizer.convert_tokens_to_ids(['[CLS]'] + x) for x in x_train]
    val_y = [data_utils.sent2input(['[PAD]'] + x, label_vocab) for x in y_train]

    model = BertForTokenClassification.from_pretrained(MODEL, num_labels=len(label_vocab))

    model = train(model, input_x, input_y, val=[val_x, val_y], outputdir=output_dir)

    #print(model)
    print()