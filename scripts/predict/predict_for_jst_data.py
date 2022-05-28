import argparse
import json
import os

import torch

from BERT.Model import NERModel
from BERT.predict import predict_from_sentences_list
from knowledge_bases.hyakuyaku import HyakuyakuList, HyakuyakuDrugIOBMatcher
from util import iob_util


def predict_file(file, output_file, model):
    matcher = HyakuyakuDrugIOBMatcher(HyakuyakuList(), 'M')
    output_file.write("<articles>\n")
    i = 0
    processed = 0
    for line in file:
        doc = json.loads(line)
        print("Article ", i, "- Skipped", i - processed, end='\r')
        i += 1

        try:
            # keywords = doc['タイトル切り出し語(絞り込み用)']
            # if not keywords or '症例' not in keywords:
            #     continue

            text = doc['文献抄録(和文)']
            if not text:
                continue
        except KeyError:
            continue

        try:
            sentences, labels = predict_from_sentences_list(model, [text], True)
            tagged_sentences = list()
            for sent, label in zip(sentences, labels):
                tagged_sentence = iob_util.convert_iob_to_xml(sent, label)
                tagged_sentence = matcher.match(tagged_sentence)
                tagged_sentences.append(tagged_sentence)

                # validate
                try:
                    iob_util.convert_xml_to_taglist(tagged_sentence, 'C')
                    iob_util.convert_xml_to_taglist(tagged_sentence, 'M')
                except:
                    print('failed\n')
                    print(tagged_sentence)

            output_file.write("<article id=\"{}\">\n".format(i))
            output_file.write("\n".join(tagged_sentences))
            output_file.write("\n</article>\n")
            processed += 1
        except Exception as e:
            print('failed')
            print(e)
    output_file.write("</articles>\n")
    output_file.flush()
    print("Processed", processed, "out of", i, "articles")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict jst data')
    parser.add_argument('--model', type=str, help='Model', required=True)
    parser.add_argument('--input', type=str, nargs="+", help='Input files', required=True)
    parser.add_argument('--output', type=str, help='Output folder', required=True)
    args = parser.parse_args()

    # Load BERT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model_name = 'cl-tohoku/bert-base-japanese-char-v2'
    model = NERModel.load_transformers_model(model_name, args.model, device)

    # Load files
    # Get file list
    # DIRECTORY = "/Users/gabriel-he/Documents/JST data/2022-05/filtered"
    # output_dir = "/Users/gabriel-he/Documents/JST data/2022-05/out"
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # file_list = sorted(glob.glob(DIRECTORY + '/[!~]*.json'))
    # file_list = ['/Users/gabriel-he/Documents/JST data/2022-05/filtered/症例_患者_治療_診断_filtered.json']
    file_list = sorted(args.input)

    for i in range(len(file_list)):
        file = file_list[i]
        print('\nFile', i + 1, 'of', len(file_list))
        print(file)

        output_filename = output_dir + "/" + file.split("/")[-1].replace("json", "txt")
        output_file = open(output_filename, "w")
        print("Output file:", output_filename)

        predict_file(open(file), output_file, model)
