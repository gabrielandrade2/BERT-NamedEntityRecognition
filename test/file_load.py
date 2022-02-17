import pandas as pd
from lxml.etree import XMLSyntaxError

from BERT.train import train_from_sentences_tags_list
from util.iob_util import convert_xml_to_iob
from util.text_utils import *


def convert_to_iob_list(texts, tag_list, should_split_sentences=False, ignore_mismatch_tags=True):
    # Preprocess
    texts = split_sentences(texts, True)
    texts = preprocessing(texts, True)

    # Convert <C> tags into </d>
    for i in range(len(texts)):
        text = texts[i]
        text = text.replace('<C>', '<d>')
        text = text.replace('<C ', '<d ')
        text = text.replace('</C>', '</d>')
        texts[i] = text

    # Convert
    items = list()
    tags = list()
    i = 0
    for t in texts:
        sent = list()
        tag = list()
        try:
            iob = convert_xml_to_iob(t, tag_list, ignore_mismatch_tags=ignore_mismatch_tags)
            # Convert tuples into lists
            for item in iob:
                if item[0] == ' ':
                    continue
                sent.append(item[0])
                tag.append(item[1])
            items.append(sent)
            tags.append(tag)
        except XMLSyntaxError:
            print("Skipping text with xml syntax error, id: " + str(i))
            print(t)
        i = i + 1
    return items, tags


if __name__ == '__main__':
    file = '../data/DATA_IM_v6.txt'
    data = pd.read_csv(file, sep="	")
    texts_tagged = data['text_tagged'].tolist()
    texts_raw = data['text_raw'].tolist()

    tag_list = ['C']
    model = 'cl-tohoku/bert-base-japanese-char-v2'

    sentences, tags = convert_to_iob_list(texts_tagged, tag_list, should_split_sentences=True)
    model = train_from_sentences_tags_list(sentences, tags, model, '../BERT/out_IM_v6')
