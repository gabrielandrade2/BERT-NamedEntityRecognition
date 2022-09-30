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
    parser.add_argument('--validation_ratio', type=float, help='Ratio of training data used in validation',
                        required=False)
    parser.add_argument('--test_file', type=str, help='Test file path', default=None)
    parser.add_argument('--test_ratio', type=float,
                        help='Ratio of training data used for testing if not test file is provided', default=0.2)
    parser.add_argument('--output', type=str, help='Output folder', required=False)
    parser.add_argument('--tags', type=str, nargs='+', help='XML tags', required=True)
    parser.add_argument('--attr', type=str, nargs='+', help='XML tag attributes', required=False, default=None)
    parser.add_argument('--split_sentences', action=argparse.BooleanOptionalAction, help='Should split sentences')
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction,
                        help='Use transformers local files')
    args = parser.parse_args()

    model_type = 'cl-tohoku/bert-base-japanese-char-v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NERModel.load_transformers_model(model_type, args.model_path, device, args.local_files_only)

    # Set training parameters
    parameters = TrainingParameters()
    # parameters.set_max_epochs(20)
    # parameters.set_learning_rate(1e-5)
    # parameters.set_batch_size(4)

    # Load the training dile
    sentences, tags = convert_xml_file_to_iob_list(args.training_file, args.tags, attr_list=args.attr,
                                                   should_split_sentences=args.split_sentences)

    # Check if a test file is provided
    if args.test_file is None:
        train_x, test_x, train_y, test_y = train_test_split(sentences, tags, test_size=args.test_ratio)
    else:
        test_x, test_y = convert_xml_file_to_iob_list(args.test_file, args.tags, attr_list=args.attr,
                                                      should_split_sentences=args.split_sentences)
        train_x = sentences
        train_y = tags

    # Finetune the model
    if args.validation_ratio is not None:
        model = finetune_from_sentences_tags_list(train_x, train_y, model, args.output, parameters=parameters,
                                                  validation_ratio=args.validation_ratio)
    else:
        model = finetune_from_sentences_tags_list(train_x, train_y, model, args.output, parameters=parameters)

    # Evaluate the model
    evaluate(model, test_x, test_y, save_dir=args.output)
