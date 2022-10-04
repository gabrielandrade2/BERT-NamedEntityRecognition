import pandas as pd
from sklearn.model_selection import train_test_split

from BERT.Model import TrainingParameters
from BERT.evaluate import evaluate
from BERT.train import train_from_sentences_tags_list
from util.iob_util import convert_xml_text_list_to_iob_list
from util.text_utils import *

if __name__ == '__main__':
    file = '../../data/DATA_IM_v6.txt'
    data = pd.read_csv(file, sep="	")
    texts_tagged = data['text_tagged'].tolist()[:1000]
    texts_raw = data['text_raw'].tolist()[:1000]

    tag_list = ['C']
    attr_list = ['MOD']
    model = 'cl-tohoku/bert-base-japanese-char-v2'

    # Preprocess
    texts = split_sentences(texts_tagged, True)
    texts = preprocessing(texts, True)

    sentences, tags, _ = convert_xml_text_list_to_iob_list(texts, tag_list, attr_list, ignore_mismatch_tags=True,
                                                           print_failed_sentences=True)

    train_x, test_x, train_y, test_y = train_test_split(sentences, tags, test_size=0.2)

    parameters = TrainingParameters()
    parameters.set_max_epochs(10)
    # parameters.set_learning_rate(1e-5)
    # parameters.set_optimizer(optim.SGD)

    model = train_from_sentences_tags_list(train_x, train_y, model, '../../out/out_IM_v6_negative_222222',
                                           device="cuda:1", parameters=parameters)

    evaluate(model, test_x, test_y)
