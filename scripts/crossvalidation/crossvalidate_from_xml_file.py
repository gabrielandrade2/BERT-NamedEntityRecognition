import argparse
import os

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from BERT.Model import TrainingParameters
from BERT.evaluate import evaluate
from BERT.train import train_from_sentences_tags_list
from util.xml_parser import convert_xml_file_to_iob_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crossvalidate from XML file')
    parser.add_argument('--training_file', type=str, help='Training file path', required=True)
    parser.add_argument('--folds', type=int, help='Number of crossvalidation fold', required=True)
    parser.add_argument('--output', type=str, help='Output folder', required=False)
    parser.add_argument('--test_file', type=str, help='Test file path', default=None)
    parser.add_argument('--test_ratio', type=float,
                        help='Ratio of training data used for testing if not test file is provided', default=0.2)
    parser.add_argument('--tags', type=str, nargs='+', help='XML tags', required=True)
    parser.add_argument('--attr', type=str, nargs='+', help='XML tag attributes', required=False, default=None)
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction,
                        help='Use transformers local files')
    parser.add_argument('--split_sentences', action=argparse.BooleanOptionalAction, help='Should split sentences')
    parser.add_argument('--device', type=str, help='Device', required=False, default="cpu")
    TrainingParameters.add_parser_arguments(parser)
    args = parser.parse_args()

    model_type = 'cl-tohoku/bert-base-japanese-char-v2'

    # Load the training file
    sentences, tags = convert_xml_file_to_iob_list(args.training_file, args.tags, attr_list=args.attr,
                                                   should_split_sentences=args.split_sentences)

    sentences = sentences[1000]
    tags = tags[1000]

    # Check if a test file is provided
    if args.test_file is None:
        train_x, test_x, train_y, test_y = train_test_split(sentences, tags, test_size=args.test_ratio)
    else:
        test_x, test_y = convert_xml_file_to_iob_list(args.test_file, args.tags, attr_list=args.attr,
                                                      should_split_sentences=args.split_sentences)
        train_x = sentences
        train_y = tags

    # Set training parameters
    parameters = TrainingParameters.from_args(args)

    # Crossvalidation
    kfold = KFold(n_splits=args.folds, shuffle=True)
    cross_val_step = 0
    train_metrics = []
    test_metrics = []
    for train_index, test_index in tqdm(kfold.split(sentences, tags), total=args.folds, desc="CV Fold"):
        cross_val_folder = os.path.join(args.output, 'crossvalidation_' + str(cross_val_step))

        train_x, val_x = [sentences[i] for i in train_index], [sentences[i] for i in test_index]
        train_y, val_y = [tags[i] for i in train_index], [tags[i] for i in test_index]

        model = train_from_sentences_tags_list(train_x, train_y, val_x, val_y, model_type,
                                               cross_val_folder,
                                               parameters=parameters,
                                               local_files_only=args.local_files_only,
                                               device=args.device)
        train_metrics.append(model.get_training_metrics())

        # Evaluate the model
        test_metrics.append(evaluate(model, test_x, test_y, save_dir=cross_val_folder, print_report=False))

        cross_val_step += 1

    # Print the metrics
    print("Training metrics:")
    # print(dict_mean(train_metrics))

    print("Test metrics:")
    # print(dict_mean(test_metrics))
