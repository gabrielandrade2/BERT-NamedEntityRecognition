import argparse
import itertools
import json
import os

import torch
from torch.multiprocessing import Pool, set_start_method

from BERT.Model import NERModel
from BERT.predict import predict_from_sentences_list
from knowledge_bases.hyakuyaku import HyakuyakuList, HyakuyakuDrugIOBMatcher
from util import iob_util


def func(line, model, matcher):
    doc = json.loads(line)

    try:
        # keywords = doc['タイトル切り出し語(絞り込み用)']
        # if not keywords or '症例' not in keywords:
        #     continue

        text = doc['文献抄録(和文)']
        if not text:
            return
    except KeyError:
        return

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

        ret = "\n".join(tagged_sentences)
        return ret

    except Exception as e:
        print('failed')
        print(e)


def predict_file(file, output_filename, model):
    matcher = HyakuyakuDrugIOBMatcher(HyakuyakuList(), 'M')

    i = 0
    with Pool(processes=10, maxtasksperchild=100) as pool:
        while True:
            # make a list of num_chunks chunks
            lines = itertools.islice(file, 1000)
            if lines:
                out = pool.starmap(func, zip(lines, itertools.repeat(model), itertools.repeat(matcher)), chunksize=8)
                out = list(filter(bool, out))
                print('out')
                print(bool(out))
                if not out:
                    break
                with open(output_filename, "a") as output_file:
                    for s in out:
                        i += 1
                        output_file.write("<article id=\"{}\">\n".format(i))
                        output_file.write(''.join(s))
                        output_file.write("\n</article>\n")
                print('Processed {} articles'.format(i))
            else:
                break

    with open(output_filename, "a") as output_file:
        output_file.write("</articles>\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict jst data')
    parser.add_argument('--model', type=str, help='Model')
    parser.add_argument('--input', type=str, nargs="+", help='Input files')
    parser.add_argument('--output', type=str, help='Output folder')
    args = parser.parse_args()

    try:
        set_start_method('forkserver')
    except RuntimeError:
        pass

    # Load BERT model
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(device)
    model_name = 'cl-tohoku/bert-base-japanese-char-v2'
    model = NERModel.load_transformers_model(model_name, args.model, device)
    model.model.share_memory()

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

        output_filename = output_dir + "/" + file.split("/")[-1].replace(".json", "_parallel.txt")
        # output_file = open(output_filename, "w")
        with open(output_filename, "w") as output_file:
            output_file.write("<articles>\n")

        print("Output file:", output_filename)

        predict_file(open(file), output_filename, model)
