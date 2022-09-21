import argparse

import torch

from BERT.Model import NERModel, TrainingParameters
from BERT.train import finetune_from_xml_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune from XML file')
    parser.add_argument('--model_path', type=str, help='Model folder', required=True)
    parser.add_argument('--training_file', type=str, help='Training file path', required=True)
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

    finetune_from_xml_file(args.training_file, model, args.tags, args.output, parameters, args.attr)
