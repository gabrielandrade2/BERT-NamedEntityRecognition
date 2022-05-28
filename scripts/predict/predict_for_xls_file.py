import argparse
import os
import re

import pandas as pd
import torch

from BERT.Model import NERModel
from BERT.predict import *
from util import iob_util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict jst data')
    parser.add_argument('--model', type=str, help='Model')
    parser.add_argument('--input', type=str, nargs="+", help='Input files')
    parser.add_argument('--output', type=str, help='Output folder')
    args = parser.parse_args()

    # Load BERT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model_name = 'cl-tohoku/bert-base-japanese-char-v2'
    model = NERModel.load_transformers_model(model_name, args.model, device)

    # Load files
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(args.input)

    should_normalize_entities = True

    for i in range(len(file_list)):
        file = file_list[i]

        print('\nFile', i + 1, 'of', len(file_list))
        print(file)

        output_filename = output_dir + "/" + file.split("/")[-1].replace("xlsx*", "txt")
        output_file = open(output_filename, "w")
        print("Output file:", output_filename)
        output_file.write("<articles>\n")

        xls = pd.ExcelFile(file)
        sheetX = xls.parse(0)

        # Get relevant columns
        try:
            texts = sheetX['S']
            drugs = sheetX['レジメン名']
        except KeyError:
            print("Sheet not found, Skipping file")
            continue

        # Skip the first item as it is the 例 line
        for text_num in range(1, len(texts)):
            print('Text', text_num + 1, 'of', len(texts), end='\r')

            text = texts[text_num]
            # Skip empty texts
            if text != text:
                continue

            # Add \n after "。" which do not already have it
            text = re.sub('。(?=[^\n])', "。\n", text)

            # Apply the model to extract symptoms
            sentences = text.split('\n')
            sentences, labels = predict_from_sentences_list(model, sentences)

            tagged_sentences = list()
            for sent, label in zip(sentences, labels):
                tagged_sentences.append(iob_util.convert_iob_to_xml(sent, label))
            output_file.write("<article id=\"{}\">\n".format(i))
            output_file.write("\n".join(tagged_sentences))
            output_file.write("\n</article>\n")

        output_file.write("</articles>\n")
        output_file.flush()
        output_file.close()
