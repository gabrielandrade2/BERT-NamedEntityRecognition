import pandas as pd
import torch

from BERT.predict import predict, convert_prediction_to_labels, remove_label_padding
from BERT.util.bert_utils import load_model
from util import iob_util, text_utils


def predict_list(texts, filename):
    output_file = open(filename, 'w')
    output_file.write("<articles>\n")
    split_texts = text_utils.split_sentences(texts, False)
    i = 0
    for text in split_texts:
        percent = '({:.2f}%)'.format(i/len(split_texts) * 100)
        print("Text ", i, "of", len(split_texts), percent, '-', len(text), end='\r')
        try:
            tokenized_texts = [tokenizer.tokenize(t) for t in text]
            sentences_embeddings = [tokenizer.convert_tokens_to_ids(['[CLS]'] + t) for t in tokenized_texts]
            tags = predict(model, sentences_embeddings, device=device)
            labels = convert_prediction_to_labels(tags, vocabulary)
            sentences = [tokenizer.convert_ids_to_tokens(t)[1:] for t in sentences_embeddings]
            labels = remove_label_padding(sentences, labels)

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
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    model, tokenizer, vocabulary = load_model(MODEL, '../../out_IM_v6')

    # Load files
    input_file = "../../data/yc_統合_2014_2020.csv"
    csv = pd.read_csv(input_file, sep=',')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Questions')
    predict_list(csv['question'].to_list(), "../../out/yc_統合_2014_2020_questions.xml")
    print('Answers')
    predict_list(csv['best_answer'].to_list(), "../../out/yc_統合_2014_2020_answers.xml")