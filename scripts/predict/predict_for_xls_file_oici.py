import argparse
import os
import re

import pandas as pd
import torch
from tqdm import tqdm

from BERT.Model import NERModel
from BERT.predict import *
from knowledge_bases.hyakuyaku import HyakuyakuList, HyakuyakuDrugMatcher
from util import iob_util
from util.text_utils import tag_matches

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict OICI data')
    parser.add_argument('--model', type=str, help='Model')
    parser.add_argument('--input', type=str, nargs="+", help='Input files')
    parser.add_argument('--output', type=str, help='Output folder')
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

    # database = MedDRADatabase('/Users/sociocom/Documents/meddra-sqlite/db/meddra.sqlite3')
    # database.open_connection()
    # normalization_model = MedDRAPatientFriendlyPTEntityNormalizer(
    #     database,
    #     MedDRAPatientFriendlyList('/Users/sociocom/Documents/MedDRA/patient-friendly_term_list_v24.1_J.xlsx')
    # )
    #
    # # Add drug english names
    # hyakuyaku = HyakuyakuList()
    # normalizer = HyakuyakuNormalizer(hyakuyaku)

    for column in ['記事テキスト', 'S', 'O', 'A', 'P']:
        for i in range(len(file_list)):
            file = file_list[i]

            print('\nFile', i + 1, 'of', len(file_list))
            print(file)

            output_filename = output_dir + "/" + column + "_" + args.model.split('/')[-1] + ".txt"
            output_file = open(output_filename, "w")
            print("Output file:", output_filename)
            output_file.write("<articles>\n")

            xls = pd.ExcelFile(file)
            sheetX = xls.parse(0)

            symptoms = list()
            symptoms_negative = list()
            drugs_list = list()
            article_ids = list()

            # Get relevant columns
            try:
                patient_ids = sheetX['患者ID']
                texts = sheetX[column]
                drugs = sheetX['レジメン名']
            except KeyError:
                print("Sheet not found, Skipping file")
                continue

            # Skip the first item as it is the 例 line
            for text_num in tqdm(range(0, len(texts))):
                text = str(texts[text_num])
                patient_id = patient_ids[text_num]
                try:
                    # print('Text', text_num + 1, 'of', len(texts), end='\r')

                    # Skip empty texts
                    if text != text:
                        article_ids.append(text_num + 1)
                        symptoms.append(list())
                        drugs_list.append(list())
                        continue

                    # Add \n after "。" which do not already have it
                    text = re.sub('。(?=[^\n])', "。\n", text)

                    # Apply the model to extract symptoms
                    sentences = text.split('\n')
                    sentences, labels = predict_from_sentences_list(model, sentences)

                    tagged_sentences = list()
                    c = []
                    cn = []
                    for sent, label in zip(sentences, labels):
                        label = [l if l != "[PAD]" else "O" for l in label]
                        tagged_sentences.append(iob_util.convert_iob_to_xml(sent, label))
                        entries = iob_util.convert_iob_to_dict(sent, label)
                        for entry in entries:
                            if entry['type'] == 'C':
                                c.append(entry['word'])
                            elif entry['type'] == 'CN':
                                cn.append(entry['word'])
                    output_file.write("<article id=\"{}\" patient_id=\"{}\">\n".format(text_num + 1, patient_id))
                    output_file.write("\n".join(tagged_sentences))
                    output_file.write("\n</article>\n")

                    # symptoms.append(
                    #     [item[3] for item in iob_util.convert_xml_to_taglist("\n".join(tagged_sentences), 'C')[1]])
                    # normalized_drug, _ = normalizer.normalize(drugs[text_num])
                    # drugs_list.append([hyakuyaku.append_english_name(normalized_drug)])
                    article_ids.append(text_num + 1)
                    symptoms.append(c)
                    symptoms_negative.append(cn)
                    drugs_list.append([drugs[text_num]])
                except Exception as e:
                    print('failed')
                    print(e)
                    print(text)
                    print('\n\n\n')

            a = pd.DataFrame(list(zip(article_ids, drugs_list, symptoms, symptoms_negative)),
                             columns=['article_id', 'レジメン名', 'C', 'CN'])
            a.to_excel(args.output + '/extracted_data_' + column + '_' + args.model.split('/')[-1] + '.xlsx',
                       index=False)

            output_file.write("</articles>\n")
            output_file.flush()
            output_file.close()

            # drugs_list = [['' if pd.isnull(drug) else drug for drug in temp] for temp in drugs_list]
            #
            # table = ade_table.from_lists(drugs_list, symptoms, normalization_model)
            # table.to_excel('/Users/sociocom/PycharmProjects/BERT-NamedEntityRecognition/out/oici/ade_table_DATA_IM_V6.xlsx')
