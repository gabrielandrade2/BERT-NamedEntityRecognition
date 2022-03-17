import glob
import json
import os

from knowledge_bases.hyakuyaku import *
from util import text_utils
from util.text_utils import tag_matches


def predict_file(file, output_file):
    matcher = HyakuyakuDrugMatcher(HyakuyakuList())
    output_file.write("<articles>\n")
    i = 0
    processed = 0
    for line in file:
        doc = json.loads(line)
        print("Article ", i, "- Skipped", i - processed, end='\r')
        i = i + 1

        try:
            keywords = doc['タイトル切り出し語(絞り込み用)']
            if not keywords or '症例' not in keywords:
                continue

            text = doc['文献抄録(和文)']
            if not text:
                continue
        except KeyError:
            continue

        processed = processed + 1
        try:
            sentences = text_utils.split_sentences([text], True)
            tagged_sentences = list()
            for sentence in sentences:
                matches = matcher.match(sentence)
                tagged_sentences.append(tag_matches(sentence, matches, 'M'))
            output_file.write("<article id=\"{}\">\n".format(i))
            output_file.write("\n".join(tagged_sentences))
            output_file.write("\n</article>\n")
        except Exception as e:
            print('failed')
            print(e)
    output_file.write("</articles>\n")
    output_file.flush()
    print("Processed", processed, "out of", i, "articles")


if __name__ == '__main__':
    # Load files
    # Get file list
    DIRECTORY = "/Users/gabriel-he/PycharmProjects/NER/data/JST data"
    output_dir = "../out/JST data-drugname"
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(DIRECTORY + '/[!~]*.json'))

    for i in range(len(file_list)):
        file = file_list[i]
        print('\nFile', i + 1, 'of', len(file_list))
        print(file)

        output_filename = output_dir + "/" + file.split("/")[-1].replace("json", "txt")
        output_file = open(output_filename, "w")
        print("Output file:", output_filename)

        predict_file(open(file), output_file)
