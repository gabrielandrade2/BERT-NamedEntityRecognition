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
    attr_list = ['MOD']
    model = 'cl-tohoku/bert-base-japanese-char-v2'

    # Preprocess
    texts = split_sentences(texts_tagged, True)
    texts = preprocessing(texts, True)

    sentences, tags, _ = convert_xml_text_list_to_iob_list(texts, tag_list, attr_list, ignore_mismatch_tags=True,
                                                           print_failed_sentences=True)

    # sentences = [[mojimoji.han_to_zen(x) for x in s] for s in sentences]
    model = train_from_sentences_tags_list(sentences, tags, model, '../../out/out_IM_v6_negative')
