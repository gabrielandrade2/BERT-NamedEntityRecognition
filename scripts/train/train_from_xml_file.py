import argparse

from BERT.train import train_from_xml_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train from XML file')
    parser.add_argument('--training_file', type=str, help='Training file path', required=True)
    parser.add_argument('--output', type=str, help='Output folder', required=False)
    parser.add_argument('--tags', type=str, nargs='+', help='XML tags', required=True)
    parser.add_argument('--attr', type=str, nargs='+', help='XML tag attributes', required=False, default=None)
    parser.add_argument('--local_files_only', type=bool, help='Use transformers local files', required=False,
                        default=False)
    args = parser.parse_args()

    model_type = 'cl-tohoku/bert-base-japanese-char-v2'
    train_from_xml_file(args.training_file, model_type, args.tags, args.output, args.attr, args.local_files_only)
