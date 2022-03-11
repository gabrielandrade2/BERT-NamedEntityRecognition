import glob
import os

import mojimoji
import pandas as pd

from BERT.Model import NERModel
from BERT.predict import predict_from_sentences_list
from util.iob_util import convert_iob_to_xml, convert_iob_to_dict

if __name__ == '__main__':
    # Load the model and file list
    directory = "../data/Croudworks薬歴/"
    file_list = glob.glob(directory + '*.xlsx')

    # load model
    model_name = 'cl-tohoku/bert-base-japanese-char-v2'
    model = NERModel.load_transformers_model(model_name, '../../out')

    MAX_LENGTH = 512

    # iterate through files
    for file in file_list:
        filename = os.path.split(file)[1]
        path = directory + filename
        print("FILE: " + path)
        xls = pd.ExcelFile(path)
        sheetX = xls.parse(0)
        try:
            texts = sheetX['患者像と薬歴（SOAPのS）']
        except KeyError:
            print("Sheet not found, Skipping file")
            continue

        # Clear empty lines
        i = 0
        for text in texts:
            if text != text:
                texts.pop(i)
            i = i + 1

        # Convert text to NFKC standard
        texts = [mojimoji.han_to_zen(t, kana=False) for t in texts]

        # Tokenize text for BERT
        data_x, labels = predict_from_sentences_list(model, texts)

        # for x, t in zip(data_x, labels):
        #     print('\n'.join([x1 + '\t' + str(x2) for x1, x2 in zip(x, t)]))

        results = list()
        for x in zip(data_x, labels):
            try:
                l = x[1]
                # print(len(l))
                end = len(x[0])
                # print(end)
                xml = convert_iob_to_xml(x[0], l[:end])
                results.append(convert_iob_to_dict(x[0], l[:end]))
                print(xml)
            except Exception as e:
                print("failed")
                print(e.message)

        # Generate a dataframe to be saved into the excel file
        i = 0
        toStoreData = []
        for text in texts:
            toPrint = ''.join(text)
            for wordsDic in results[i]:
                print(wordsDic["type"], " - ", wordsDic["word"])
                entry = {
                    "id": i + 1,
                    "text": toPrint,
                    "type": wordsDic["type"],
                    "drug": wordsDic["word"]
                }
                toStoreData.append(entry)
                toPrint = ''
            i = i + 1

        # Save the dataframe to the file
        df = pd.DataFrame(toStoreData, columns=['id', 'text', 'type', 'drug'])
        with pd.ExcelWriter(path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='my_extracted_drugs')
        print("")
