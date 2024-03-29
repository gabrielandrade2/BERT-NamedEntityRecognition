import argparse

import torch
from tqdm import tqdm

from BERT.Model import NERModel
from BERT.predict import predict_from_sentences_list
from util.iob_util import convert_list_iob_to_xml
from util.xml_parser import xml_to_articles, Article, articles_to_xml


def main(model_path, input_file, output_file, split_sentences, local_files_only, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_type = 'cl-tohoku/bert-base-japanese-char-v2'
    model = NERModel.load_transformers_model(model_type, model_path, device, local_files_only)

    # Load XML file
    articles = xml_to_articles(input_file)

    # Predict
    processed_articles = []
    for article in tqdm(articles, desc='Predicting', total=len(articles), ascii=True, ncols=100):
        sentences, iob = predict_from_sentences_list(model, article.text, split_sentences=split_sentences,
                                                     display_progress=False)
        processed_text = convert_list_iob_to_xml(sentences, iob)
        processed_articles.append(Article(processed_text, article.headers))

    # Save to XML file
    articles_to_xml(processed_articles, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict from XML file')
    parser.add_argument('--model_path', type=str, help='Model folder', required=True)
    parser.add_argument('--input_file', type=str, help='Input file path', default=None)
    parser.add_argument('--output_file', type=str, help='Output file path', default=None)
    parser.add_argument('--split_sentences', action=argparse.BooleanOptionalAction, help='Should split sentences')
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction,
                        help='Use transformers local files')
    parser.add_argument('--device', type=str, help='Device to run model on', default=None, required=False)
    args = parser.parse_args()

    main(args.model_path, args.input_file, args.output_file, args.split_sentences, args.local_files_only, args.device)
