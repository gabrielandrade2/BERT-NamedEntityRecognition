import os

from util.bert.xml_parser import convert_xml_to_iob_file

if __name__ == '__main__':

    xmlFile = '../data/MedTxt-CR-JA-training-v2.xml'
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    TAG_LIST = ['d']
    output_dir = '../BERT/out'

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print("folder exists")

    ##### Load the data #####
    convert_xml_to_iob_file(xmlFile, TAG_LIST, 'test.iob')

    # Get articles from file
    # xmlFile = '../data/drugHistoryCheck.xml'
    # xmlFile = '../data/MedTxt-CR-JA-training-v2.xml'
    # texts = xml_to_articles(xmlFile)
    # texts = __preprocessing(texts)
    # texts = split_sentences(texts)
    # texts = drop_texts_with_mismatched_tags(texts)
    #
    # # Get iob info from xml
    # original_sentences, original_labels = convert_xml_to_iob_list(xmlFile, TAG_LIST, split_sentences=True)