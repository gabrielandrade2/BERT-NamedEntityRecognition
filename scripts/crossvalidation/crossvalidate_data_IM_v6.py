import os

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

from BERT.Model import TrainingParameters
from BERT.evaluate import evaluate
from BERT.train import train_from_sentences_tags_list_val
from scripts.crossvalidation.utils import crossvalidation_utils
from util.iob_util import convert_xml_text_list_to_iob_list
from util.text_utils import *

if __name__ == '__main__':
    file = 'data/DATA_IM_v6.txt'
    data = pd.read_csv(file, sep="	")
    texts_tagged = data['text_tagged'].tolist()
    texts_raw = data['text_raw'].tolist()

    tag_list = ['C']
    attr_list = ['MOD']
    model_type = 'cl-tohoku/bert-base-japanese-char-v2'

    # Preprocess
    texts = split_sentences(texts_tagged, True)
    texts = preprocessing(texts, True)

    sentences, tags, _ = convert_xml_text_list_to_iob_list(texts, tag_list, attr_list, ignore_mismatch_tags=True,
                                                           print_failed_sentences=True)

    train_x, test_x, train_y, test_y = train_test_split(sentences, tags, test_size=0.2)

    # Set training parameters
    parameters = TrainingParameters()

    # Crossvalidation
    kfold = KFold(n_splits=10, shuffle=True)
    cross_val_step = 0
    train_metrics = []
    test_metrics = []
    for train_index, test_index in tqdm(kfold.split(sentences, tags), total=10, desc="CV Fold", position=0, leave=True):
        cross_val_folder = os.path.join('out/IM_v6_crossval', 'crossvalidation_' + str(cross_val_step))

        train_x, val_x = [sentences[i] for i in train_index], [sentences[i] for i in test_index]
        train_y, val_y = [tags[i] for i in train_index], [tags[i] for i in test_index]

        model = train_from_sentences_tags_list_val(train_x, train_y, val_x, val_y, model_type,
                                                   cross_val_folder,
                                                   parameters=parameters,
                                                   local_files_only=True,
                                                   device="cuda:1")
        train_metrics.append(model.get_training_metrics())

        # Evaluate the model
        test_metrics.append(evaluate(model, test_x, test_y, save_dir=cross_val_folder, print_report=False))

        cross_val_step += 1

    # Print the metrics
    crossvalidation_utils.average_training_metrics(train_metrics, save_dir='out/IM_v6_crossval')
    crossvalidation_utils.average_test_metrics(test_metrics, save_dir='out/IM_v6_crossval')
