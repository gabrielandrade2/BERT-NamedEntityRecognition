import argparse

import ade_table
from knowledge_bases.meddra import MedDRADatabase, MedDRAPatientFriendlyPTEntityNormalizer, MedDRAPatientFriendlyList
from util import xml_parser, iob_util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ADE table from tagged xml file')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--generate_heatmap', type=bool, help='Generate heatmap picture', default=False)
    args = parser.parse_args()

    file = open(args.input_file)
    articles = xml_parser.xml_to_articles(file);
    symptoms = list()
    drugs = list()

    for article in articles:
        symptoms.append([item[3] for item in iob_util.convert_xml_to_taglist(article, 'C')[1]])
        drugs.append([item[3] for item in iob_util.convert_xml_to_taglist(article, 'M')[1]])

    database = MedDRADatabase('/Users/gabriel-he/Documents/git/meddra-sqlite/db/meddra.sqlite3')
    database.open_connection()
    normalization_model = MedDRAPatientFriendlyPTEntityNormalizer(
        database,
        MedDRAPatientFriendlyList('/Users/gabriel-he/Documents/MedDRA/patient-friendly_term_list_v24.1_J.xlsx')
    )

    table = ade_table.from_lists(drugs, symptoms, normalization_model)

    output_path = args.input_file.replace('.xml', '.xlsx')
    table.to_dataframe().to_excel(output_path)

    if args.generate_heatmap:
        table.generate_heatmap(output_path=output_path.replace('.xlsx', '.png'))
