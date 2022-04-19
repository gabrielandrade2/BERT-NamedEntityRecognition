import unicodedata

import pandas as pd
from seqeval.metrics import accuracy_score, precision_score, f1_score, classification_report
from seqeval.scheme import IOB2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from BERT.predict import predict_from_sentences_list
from BERT.train import train_from_xml_texts
from util import iob_util, text_utils, list_utils
from util.Dataset import YakurekiTxtDataset
from util.list_utils import flatten_list
from util.xlarge import score

if __name__ == '__main__':
    dataset = YakurekiTxtDataset("/Users/gabriel-he/Documents/datasets/薬歴/薬歴_タグ付け済_中江")
    texts = [text for text in dataset]
    train, test = train_test_split(texts, test_size=0.3)

    taglist = ['d']
    model = train_from_xml_texts(train, 'cl-tohoku/bert-base-japanese-char-v2', taglist, '../out/yakurekimodel')
    # model = NERModel.load_transformers_model('cl-tohoku/bert-base-japanese-char-v2', '../out/yakurekimodel')

    gold_tags = list()
    predicted_tags = list()
    count = 0

    output = open('output.txt', 'w')
    output_data = list()

    for tagged_text in tqdm(test):
        tagged_text = unicodedata.normalize('NFKC', tagged_text)
        tagged_sentences = tagged_text.split('\n')

        gold = iob_util.convert_xml_text_list_to_iob_list(tagged_sentences, 'd', ignore_mismatch_tags=False)
        gold = model.normalize_tagged_dataset(gold[0], gold[1])

        sentences = text_utils.remove_tags(tagged_text).split('\n')
        predicted = predict_from_sentences_list(model, sentences)

        # Convert d to C for comparison
        # predicted_temp = [[tag.replace('C', 'd') for tag in tag_list] for tag_list in predicted[1]]
        predicted_temp = predicted[1]
        gold_temp = gold[1]

        if list_utils.list_size(gold_temp) == list_utils.list_size(predicted_temp):
            predicted_tags.extend(predicted_temp)
            gold_tags.extend(gold_temp)

            ###### Calculate performance metrics #####

            xlarge_results = dict()
            row = (text_utils.remove_tags(tagged_text, tag_list=['m-key']),
                   iob_util.convert_list_iob_to_xml(predicted[0], predicted[1]),
                   score(flatten_list(gold_temp), flatten_list(predicted_temp), output_dict=xlarge_results),
                   accuracy_score(gold_temp, predicted_temp),
                   precision_score(gold_temp, predicted_temp, zero_division=0),
                   f1_score(gold_temp, predicted_temp, zero_division=0),
                   xlarge_results["exact_match"],
                   xlarge_results["exceeding_match"],
                   xlarge_results["exceeding_match_overlap"],
                   xlarge_results["partial_match"],
                   xlarge_results["partial_match_overlap"],
                   xlarge_results["missing_match"],
                   xlarge_results["incorrect_match"]
                   )
            output_data.append(row)

            output.write('\n')
            output.write(row[0])
            output.write('\n')
            output.write(row[1])
            output.write('\n')
            output.write(str(xlarge_results))
            output.write('\n')
            output.write('XLarge score: ' + str(row[2]))
            output.write('\n')
            output.write('Accuracy: ' + str(row[3]))
            output.write('\n')
            output.write('Precision: ' + str(row[4]))
            output.write('\n')
            output.write('F1 score: ' + str(row[5]))
            output.write('\n')
            try:
                output.write(
                    classification_report(gold_temp, predicted_temp, mode='strict', scheme=IOB2, zero_division=0))
                output.write('\n')
            except ValueError:
                pass
        else:
            count += 1

    # print('Ignored {}/{} texts'.format(count, len(dataset)))
    # iob_util.evaluate_performance(gold_tags, predicted_tags)

    output_df = pd.DataFrame(output_data, columns=['original text',
                                                   'tagged text',
                                                   'strawberry',
                                                   'accuracy',
                                                   'precision',
                                                   'f1 score',

                                                   "exact_match", "exceeding_match", "exceeding_match_overlap",
                                                   "partial_match", "partial_match_overlap",
                                                   "missing_match", "incorrect_match"]
                             )
    output_df.to_excel('output.xlsx')

    # model = NERModel.load_transformers_model('cl-tohoku/bert-base-japanese-char-v2', '../out/out_IM_v6')
    #
    # gold_tags = list()
    # predicted_tags = list()
    # count = 0
    # for tagged_text in tqdm(dataset):
    #     tagged_text = unicodedata.normalize('NFKC', tagged_text)
    #     tagged_sentences = tagged_text.split('\n')
    #     tagged_sentences = [t for t in tagged_sentences if not t.startswith('既往歴')]
    #
    #     gold = iob_util.convert_xml_text_list_to_iob_list(tagged_sentences, 'd', ignore_mismatch_tags=False)
    #     gold = model.normalize_tagged_dataset(gold[0], gold[1])
    #
    #     sentences = text_utils.remove_tags(tagged_text).split('\n')
    #     sentences = [t for t in sentences if not t.startswith('既往歴')]
    #     predicted = predict_from_sentences_list(model, sentences)
    #
    #     # Convert d to C for comparison
    #     predicted_temp = [[tag.replace('C', 'd') for tag in tag_list] for tag_list in predicted[1]]
    #     gold_temp = gold[1]
    #
    #     if list_utils.list_size(gold_temp) == list_utils.list_size(predicted_temp):
    #         predicted_tags.extend(predicted_temp)
    #         gold_tags.extend(gold_temp)
    #
    #         print(flatten_list(gold[0]))
    #         gold_temp = flatten_list(gold_temp)
    #         print(gold_temp)
    #         predicted_temp = flatten_list(predicted_temp)
    #         print(predicted_temp)
    #         print(score(gold_temp, predicted_temp))
    #     else:
    #         count += 1
    #
    # print('Ignored {}/{} texts'.format(count, len(dataset)))
    # iob_util.evaluate_performance(gold_tags, predicted_tags)
