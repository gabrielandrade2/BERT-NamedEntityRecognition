import argparse
import os
import re

import pandas as pd
import torch
from tqdm import tqdm

from BERT.Model import NERModel
from BERT.predict import *
from knowledge_bases.hyakuyaku import HyakuyakuList, HyakuyakuDrugIOBMatcher
from util import iob_util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict jst data')
    parser.add_argument('--model', type=str, help='Model')
    parser.add_argument('--input', type=str, nargs="+", help='Input files')
    parser.add_argument('--output', type=str, help='Output folder')
    parser.add_argument('--tag_drugs', type=bool, help='Should tag drugs')
    args = parser.parse_args()

    # Load BERT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model_name = 'cl-tohoku/bert-base-japanese-char-v2'
    model = NERModel.load_transformers_model(model_name, args.model, device, local_files_only=True)

    # Load files
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(args.input)

    should_normalize_entities = True

    tag_drugs = args.tag_drugs

    # database = MedDRADatabase('/Users/sociocom/Documents/meddra-sqlite/db/meddra.sqlite3')
    # database.open_connection()
    # normalization_model = MedDRAPatientFriendlyPTEntityNormalizer(
    #     database,
    #     MedDRAPatientFriendlyList('/Users/sociocom/Documents/MedDRA/patient-friendly_term_list_v24.1_J.xlsx')
    # )

    # Add drug english names
    if tag_drugs:
        hyakuyaku = HyakuyakuList(path="/Users/gabriel-he/Documents/HYAKUYAKU_FULL_v20210706.xlsx")
        matcher = HyakuyakuDrugIOBMatcher(hyakuyaku, 'M')
    # normalizer = HyakuyakuNormalizer(hyakuyaku)

    for i in range(len(file_list)):
        file = file_list[i]

        print('\nFile', i + 1, 'of', len(file_list))
        print(file)

        output_filename = output_dir + "/" + "SOAP1_" + args.model.split('/')[-1] + ".txt"
        output_file = open(output_filename, "w")
        print("Output file:", output_filename)
        output_file.write("<articles>\n")

        xls = pd.ExcelFile(file)
        sheetX = xls.parse(0)

        symptoms = list()
        drugs_list = list()

        # Get relevant columns
        try:
            patient_ids = sheetX['患者ID']
            texts = sheetX['記事テキスト']
            drugs = sheetX['レジメン名']
        except KeyError:
            print("Sheet not found, Skipping file")
            continue

        # Skip the first item as it is the 例 line
        for text_num in tqdm(range(0, 1)):
            text = texts[text_num]
            patient_id = patient_ids[text_num]
            try:
                # print('Text', text_num + 1, 'of', len(texts), end='\r')

                # Skip empty texts
                if text != text:
                    continue

                # Add \n after "。" which do not already have it
                text = re.sub('。(?=[^\n])', "。\n", text)

                # Apply the model to extract symptoms
                sentences = text.split('\n')
                sentences, labels = predict_from_sentences_list(model, sentences)

                tagged_sentences = list()
                temp_symptoms = []
                for sent, label in zip(sentences, labels):
                    label = [l if l != "[PAD]" else "O" for l in label]
                    tagged_sentence = iob_util.convert_iob_to_xml(sent, label)
                    if tag_drugs:
                        tagged_sentence = matcher.match(tagged_sentence)
                    tagged_sentences.append(tagged_sentence)
                    entries = iob_util.convert_iob_to_dict(sent, label)
                    for entry in entries:
                        temp_symptoms.append(entry['word'])
                output_file.write("<article id=\"{}\" patient_id=\"{}\">\n".format(text_num + 1, patient_id))
                output_file.write("\n".join(tagged_sentences))
                output_file.write("\n</article>\n")

                # symptoms.append(
                #     [item[3] for item in iob_util.convert_xml_to_taglist("\n".join(tagged_sentences), 'C')[1]])
                # normalized_drug, _ = normalizer.normalize(drugs[text_num])
                # drugs_list.append([hyakuyaku.append_english_name(normalized_drug)])
                symptoms.append(temp_symptoms)
                drugs_list.append([drugs[text_num]])
            except Exception as e:
                print('failed')
                print(e)
                print(text)
                print('\n\n\n')

        a = pd.DataFrame([drugs_list, symptoms]).transpose()
        a.to_excel(args.output + '/extracted_data_SOAP1_' + args.model.split('/')[-1] + '.xlsx')

        output_file.write("</articles>\n")
        output_file.flush()
        output_file.close()

        # drugs_list = [['' if pd.isnull(drug) else drug for drug in temp] for temp in drugs_list]
        #
        # table = ade_table.from_lists(drugs_list, symptoms, normalization_model)
        # table.to_excel('/Users/sociocom/PycharmProjects/BERT-NamedEntityRecognition/out/oici/ade_table_DATA_IM_V6.xlsx')
