import argparse

import torch
from sklearn.model_selection import train_test_split

from BERT.Model import NERModel, TrainingParameters
from BERT.evaluate import evaluate
from BERT.train import finetune_from_sentences_tags_list
from util.xml_parser import convert_xml_file_to_iob_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune from XML file')
    parser.add_argument('--model_path', type=str, help='Model folder', required=True)
    parser.add_argument('--training_file', type=str, help='Training file path', required=True)
    parser.add_argument('--test_file', type=str, help='Test file path', default=None)
    parser.add_argument('--output', type=str, help='Output folder', required=False)
    parser.add_argument('--tags', type=str, nargs='+', help='XML tags', required=True)
    parser.add_argument('--attr', type=str, nargs='+', help='XML tag attributes', required=False, default=None)
    parser.add_argument('--local_files_only', type=bool, help='Use transformers local files', required=False,
                        default=False)
    args = parser.parse_args()

    model_type = 'cl-tohoku/bert-base-japanese-char-v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NERModel.load_transformers_model(model_type, args.model_path, device, args.local_files_only)

    parameters = TrainingParameters()
    # parameters.set_max_epochs(30)
    parameters.set_learning_rate(5e-5)
    parameters.set_batch_size(2)

    sentences, tags = convert_xml_file_to_iob_list(args.training_file, args.tags, attr_list=args.attr)
    train_x, test_x, train_y, test_y = train_test_split(sentences, tags, test_size=0.2)
    model = finetune_from_sentences_tags_list(train_x, train_y, model, args.output, parameters=parameters)
    evaluate(model, test_x, test_y)

    # model = finetune_from_xml_file(args.training_file, model, args.tags, args.output, parameters, args.attr)
    #
    # if args.training_file is not None:
    #     sentences, tags = convert_xml_file_to_iob_list(args.training_file, args.tags, attr_list=args.attr)
    #     evaluate(model, sentences, tags)
