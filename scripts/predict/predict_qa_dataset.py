import pandas as pd
import torch

from BERT.Model import NERModel
from BERT.predict import predict_from_sentences_list
from util import iob_util, text_utils


def predict_list(texts, filename):
    output_file = open(filename, 'w')
    output_file.write("<articles>\n")
    split_texts = text_utils.split_sentences(texts, False)
    i = 0
    for text in split_texts:
        percent = '({:.2f}%)'.format(i / len(split_texts) * 100)
        print("Text ", i, "of", len(split_texts), percent, '-', len(text), end='\r')
        try:
            sentences, labels = predict_from_sentences_list(model, text)
            tagged_sentences = list()
            for sent, label in zip(sentences, labels):
                tagged_sentences.append(iob_util.convert_iob_to_xml(sent, label))
            output_file.write("<article id=\"{}\">\n".format(i))
            output_file.write("\n".join(tagged_sentences))
            output_file.write("\n</article>\n")
        except:
            pass
        i = i + 1
    output_file.write("</articles>\n")


if __name__ == '__main__':
    # Load BERT model
    model_name = 'cl-tohoku/bert-base-japanese-char-v2'
    model = NERModel.load_transformers_model(model_name, '../../out/out_IM_v6')

    # Load files
    input_file = "../../data/yc_統合_2014_2020.csv"
    csv = pd.read_csv(input_file, sep=',')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Questions')
    predict_list(csv['question'].to_list(), "../../out/yc_統合_2014_2020_questions.xml")
    print('Answers')
    predict_list(csv['best_answer'].to_list(), "../../out/yc_統合_2014_2020_answers.xml")
