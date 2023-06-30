import argparse

import torch

from BERT.Model import NERModel
from BERT.evaluate import evaluate
from util.xml_parser import convert_xml_file_to_iob_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict from XML file')
    parser.add_argument('--model_path', type=str, help='Model folder', required=True)
    parser.add_argument('--input_file', type=str, help='Input file path', default=None)
    parser.add_argument('--tags', type=str, nargs='+', help='XML tags', required=True)
    parser.add_argument('--attr', type=str, nargs='+', help='XML tag attributes', required=False, default=None)
    parser.add_argument('--split_sentences', action=argparse.BooleanOptionalAction, help='Should split sentences')
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction,
                        help='Use transformers local files')
    parser.add_argument('--save_dir', type=str, help='Path of the folder to save the metrics', default=None)
    parser.add_argument('--save_output_file', action=argparse.BooleanOptionalAction,
                        help='Should save the produced output file? (Must have provided a save folder')
    args = parser.parse_args()

    model_type = 'cl-tohoku/bert-base-japanese-char-v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NERModel.load_transformers_model(model_type, args.model_path, device, args.local_files_only)

    sentences, tags = convert_xml_file_to_iob_list(args.input_file, args.tags,
                                                   should_split_sentences=args.split_sentences, attr_list=args.attr)

    evaluate(model, sentences, tags, save_dir=args.save_dir, save_output_file=args.save_output_file)
