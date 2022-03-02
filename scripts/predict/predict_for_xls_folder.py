import glob
import os
import re
import pandas as pd

from BERT.predict import *
from BERT.util.bert_utils import load_model
from util import iob_util

if __name__ == '__main__':
    # Load BERT model
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    model, tokenizer, vocabulary = load_model(MODEL, '../../out')

    # Get file list
    DIRECTORY = "../../data/Croudworks薬歴/"
    output_dir = "../../data/Croudworks薬歴/tagged"
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    file_list = glob.glob(DIRECTORY + '[!~]*.xlsx')

    should_normalize_entities = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(len(file_list)):
        file = file_list[i]

        print('\nFile', i + 1, 'of', len(file_list))
        print(file)

        output_filename = output_dir + "/" + file.split("/")[-1].replace("xlsx", "txt")
        output_file = open(output_filename, "w+")
        print("Output file:", output_filename)

        xls = pd.ExcelFile(file)
        sheetX = xls.parse(0)

        # Get relevant columns
        try:
            texts = sheetX['患者像と薬歴（SOAPのS）']
            drugs = sheetX['薬剤名']
            # ades = sheetX['想定した有害事象']
            # locations = sheetX['想定した服薬指導実施場所（調剤薬局，病院（外来），病院（病棟））']
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
            sentences, labels = predict_from_sentences_list(sentences, model, tokenizer, vocabulary, device)

            tagged_sentences = list()
            for sent, label in zip(sentences, labels):
                tagged_sentences.append(iob_util.convert_iob_to_xml(sent, label))
            output_file.write("Text " + str(text_num) + "\n\n")
            output_file.write("\n".join(tagged_sentences))
            output_file.write("\n\n\n")

        print('')
        output_file.close()
