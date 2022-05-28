import argparse
from itertools import repeat

import ade_table
from knowledge_bases.hyakuyaku import HyakuyakuList
from knowledge_bases.meddra import MedDRADatabase, MedDRAPatientFriendlyPTEntityNormalizer, MedDRAPatientFriendlyList
from util import xml_parser, iob_util


def wrapper(drug_list, hyakuyaku):
    return [hyakuyaku.append_english_name(drug) for drug in drug_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ADE table from tagged xml file')
    parser.add_argument('--input_file', type=str, help='Input file', required=True)
    parser.add_argument('--generate_heatmap', type=bool, help='Generate heatmap picture', default=False)
    parser.add_argument('--normalize', type=bool, help='Normalize?', default=False)
    args = parser.parse_args()

    file = args.input_file
    articles = xml_parser.xml_to_articles(file, return_iterator=True)
    symptoms = list()
    drugs = list()

    # To be used in case we need to analyse relation intra-sentence
    # articles = text_utils.split_sentences(articles, True)

    i = 0
    for article in articles:
        i += 1
        try:
            symptoms.append([item[3] for item in iob_util.convert_xml_to_taglist(article, 'C')[1]])
            drugs.append([item[3] for item in iob_util.convert_xml_to_taglist(article, 'M')[1]])
        except Exception as e:
            print(i)
            print(article)
            print(e)

    if args.normalize:
        database = MedDRADatabase('/Users/gabriel-he/Documents/git/meddra-sqlite/db/meddra.sqlite3')
        database.open_connection()
        normalization_model = MedDRAPatientFriendlyPTEntityNormalizer(
            database,
            MedDRAPatientFriendlyList('/Users/gabriel-he/Documents/MedDRA/patient-friendly_term_list_v24.1_J.xlsx')
        )

        # Add drug english names
        hyakuyaku = HyakuyakuList()

        from multiprocessing import Pool

        pool = Pool()
        drugs = pool.starmap(wrapper, zip(drugs, repeat(hyakuyaku)))
        # drugs = [[hyakuyaku.append_english_name(drug) for drug in drug_list] for drug_list in tqdm(drugs)]

        table = ade_table.from_lists(drugs, symptoms, normalization_model)

    else:
        table = ade_table.from_lists(drugs, symptoms)

    output_path = args.input_file.replace('.xml', '.xlsx')
    # table.to_excel(output_path)
    table.to_dataframe().to_csv(output_path)

    if args.generate_heatmap:
        table.generate_heatmap(output_path=output_path.replace('.xlsx', '.png'))
