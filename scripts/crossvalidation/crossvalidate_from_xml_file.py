import argparse
import itertools
import os

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.multiprocessing import Pool
from tqdm import tqdm

from BERT.Model import TrainingParameters, NERModel
from BERT.evaluate import evaluate
from BERT.train import train_from_sentences_tags_list_val, finetune_from_sentences_tags_list_val
from scripts.crossvalidation.utils import crossvalidation_utils
from util.xml_parser import convert_xml_file_to_iob_list


# initialize worker processes
def init_worker(_train_x, _train_y, _test_x, _test_y):
    # declare scope of a new global variable
    global train_x, train_y, test_x, test_y
    # store argument in the global variable for this process
    train_x = _train_x
    train_y = _train_y
    test_x = _test_x
    test_y = _test_y


def main(args):
    model_type = 'cl-tohoku/bert-base-japanese-char-v2'
    global train_x, train_y, test_x, test_y

    # Load the training file
    sentences, tags = convert_xml_file_to_iob_list(args.training_file, args.tags, attr_list=args.attr,
                                                   should_split_sentences=args.split_sentences)

    # Check if a test file is provided
    if args.test_file is None:
        train_x, test_x, train_y, test_y = train_test_split(sentences, tags, test_size=args.test_ratio, shuffle=False)
    else:
        test_x, test_y = convert_xml_file_to_iob_list(args.test_file, args.tags, attr_list=args.attr,
                                                      should_split_sentences=args.split_sentences)
        train_x = sentences
        train_y = tags

    # Set training parameters
    parameters = TrainingParameters.from_args(args)

    # Crossvalidation
    kfold = list(KFold(n_splits=args.folds, shuffle=True).split(train_x, train_y))
    train_metrics = []
    test_metrics = []

    if not args.parallel:
        cross_val_step = 0
        for train_index, val_index in tqdm(kfold, total=args.folds, desc="CV Fold"):
            out = train(cross_val_step, train_index, val_index, model_type, args, parameters)
            cross_val_step += 1
            train_metrics.append(out[0])
            test_metrics.append(out[1])

    else:
        with Pool(processes=args.max_parallel_processes, initializer=init_worker,
                  initargs=(train_x, train_y, test_x, test_y)) as pool:
            train_index, val_index = [[i for i, j in kfold], [j for i, j in kfold]]
            out = pool.starmap(train, zip(range(args.folds), train_index, val_index, itertools.repeat(model_type),
                                          itertools.repeat(args), itertools.repeat(parameters)))
            train_metrics, test_metrics = out

    # Print the metrics
    crossvalidation_utils.average_training_metrics(train_metrics, args.output)
    crossvalidation_utils.average_test_metrics(test_metrics, args.output)


def train(cross_val_step, train_index, val_index, model_type, args, parameters):
    cross_val_folder = os.path.join(args.output, 'crossvalidation_' + str(cross_val_step))

    train_x_fold, val_x_fold = [train_x[i] for i in train_index], [train_x[i] for i in val_index]
    train_y_fold, val_y_fold = [train_y[i] for i in train_index], [train_y[i] for i in val_index]

    if args.model is not None:
        model = NERModel.load_transformers_model(model_type, args.model, args.device, args.local_files_only)
        model = finetune_from_sentences_tags_list_val(train_x_fold, train_y_fold, val_x_fold, val_y_fold, model,
                                                      cross_val_folder, parameters)
    else:
        model = train_from_sentences_tags_list_val(train_x, train_y, val_x_fold, val_y_fold, model_type,
                                                   cross_val_folder,
                                                   parameters=parameters,
                                                   local_files_only=args.local_files_only,
                                                   device=args.device)

    train_m = model.get_training_metrics()

    # Evaluate the model
    test_m = evaluate(model, test_x, test_y, save_dir=cross_val_folder, print_report=False)

    return (train_m, test_m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crossvalidate from XML file')
    parser.add_argument('--model', type=str, help='Optional model to load', required=False)
    parser.add_argument('--training_file', type=str, help='Training file path', required=True)
    parser.add_argument('--folds', type=int, help='Number of crossvalidation fold', required=True)
    parser.add_argument('--output', type=str, help='Output folder', required=False)
    parser.add_argument('--test_file', type=str, help='Test file path', default=None)
    parser.add_argument('--test_ratio', type=float,
                        help='Ratio of training data used for testing if not test file is provided', default=0.2)
    parser.add_argument('--tags', type=str, nargs='+', help='XML tags', required=True)
    parser.add_argument('--attr', type=str, nargs='+', help='XML tag attributes', required=False, default=None)
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction,
                        help='Use transformers local files?')
    parser.add_argument('--split_sentences', action=argparse.BooleanOptionalAction, help='Should split sentences?')
    parser.add_argument('--device', type=str, help='Device', required=False, default="cpu")
    parser.add_argument('--parallel', action=argparse.BooleanOptionalAction, help='Parallel run?', required=False)
    parser.add_argument('--max_parallel_processes', type=int, help='Number of processes to spawn', required=False)

    TrainingParameters.add_parser_arguments(parser)
    args = parser.parse_args()

    main(args)
