import argparse
import os
from pathlib import Path

import torch

from BERT.Model import NERModel
from BERT.predict import predict_from_sentences_list
from util import iob_util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict OICI data')
    parser.add_argument('--model', type=str, help='Model path', required=True)
    parser.add_argument('--input', type=str, nargs="+", help='Input files', required=True)
    parser.add_argument('--output', type=str, help='Output file', required=True)
    parser.add_argument('--device', type=str, help='Device to run model on', default=None, required=False)

    args = parser.parse_args()

    # Load BERT model
    model_name = 'cl-tohoku/bert-base-japanese-char-v2'
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = NERModel.load_transformers_model(model_name, args.model, device, local_files_only=True)

    # Load files
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, Path(args.output).stem + ".xml")
    output_file = open(output_filename, "w")
    output_file.write("<articles>\n")

    file_list = sorted(args.input)

    # Execute
    for file_num in range(len(file_list)):
        try:
            file = file_list[file_num]
            print('\nFile', file_num + 1, 'of', len(file_list))
            print(file)

            with open(file, 'r') as f:
                lines = f.readlines()
                sentences, labels = predict_from_sentences_list(model, lines, True)

                tagged_sentences = list()
                for sent, label in zip(sentences, labels):
                    tagged_sentences.append(iob_util.convert_iob_to_xml(sent, label))

            output_file.write("<article id=\"{}\" filename=\"{}\">\n".format(file_num + 1, Path(file).stem))
            output_file.write("\n".join(tagged_sentences))
            output_file.write("\n</article>\n")
        except Exception as e:
            print('Failed')
            print(e)
            print('\n\n')

    output_file.write("</articles>\n")
    output_file.flush()
    output_file.close()
