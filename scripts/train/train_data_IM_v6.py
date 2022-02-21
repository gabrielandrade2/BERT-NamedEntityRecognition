import pandas as pd

from BERT.train import train_from_sentences_tags_list
from util.iob_util import convert_xml_text_list_to_iob_list
from util.text_utils import *


if __name__ == '__main__':
    file = '../../data/DATA_IM_v6.txt'
    data = pd.read_csv(file, sep="	")
    texts_tagged = data['text_tagged'].tolist()
    texts_raw = data['text_raw'].tolist()

    tag_list = ['C']
    model = 'cl-tohoku/bert-base-japanese-char-v2'

    # Preprocess
    texts = split_sentences(texts_tagged, True)
    texts = preprocessing(texts, True)

    # Convert <C> tags into </d>
    # for i in range(len(texts)):
    #     text = texts[i]
    #     text = text.replace('<C>', '<d>')
    #     text = text.replace('<C ', '<d ')
    #     text = text.replace('</C>', '</d>')
    #     texts[i] = text

    sentences, tags = convert_xml_text_list_to_iob_list(texts, tag_list, ignore_mismatch_tags=True, print_failed_sentences=True)
    model = train_from_sentences_tags_list(sentences, tags, model, '../../BERT/out_IM_v6')
