import glob
import os

import pandas as pd
import torch

from BERT.Model import NERModel
from BERT.bert_utils import normalize_dataset
from util import iob_util

if __name__ == '__main__':
    # Load BERT model
    model = NERModel.load_transformers_model('cl-tohoku/bert-base-japanese-char-v2', '../../out/out_IM_v6')

    # Get file list
    DIRECTORY = "../../data/2021-tweet副作用/"
    output_dir = "../../out/2021-tweet副作用/"
    os.makedirs(output_dir, exist_ok=True)

    file_list = glob.glob(DIRECTORY + '[!~]*.csv')

    should_normalize_entities = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    original_sentences_list = list()
    predict_sentences_list = list()

    original_labels_list = list()
    predict_labels_list = list()

    for i in range(len(file_list)):
        file = file_list[i]

        print('\nFile', i + 1, 'of', len(file_list))
        print(file)

        output_filename = output_dir + "/" + file.split("/")[-1].replace("csv", "txt")
        output_file = open(output_filename, "w+")
        print("Output file:", output_filename)

        csv = pd.read_csv(file, sep='^([^,]+),', engine='python', header=None)
        # Get relevant columns
        texts = csv[2].to_list()

        # Skip the first item as it is the 例 line
        for text_num in range(1, len(texts)):
            print('Text', text_num + 1, 'of', len(texts), end='\r')

            text = texts[text_num]

            # Skip empty texts
            if text != text:
                continue

            import mojimoji

            text = mojimoji.han_to_zen(text, ascii=False, digit=False)

            # text = text.replace('<C>', '<d>')
            # text = text.replace('</C>', '</d>')

            t = iob_util.convert_xml_text_to_iob(text, tag_list=['C'])
            original_sentences, original_labels = map(list, zip(*t))

            # original_sentences_list.append(original_sentences)
            # original_labels_list.append(original_labels)

            l1 = list()
            l1.append(original_sentences)
            l2 = list()
            l2.append(original_labels)
            original_sentences, original_labels = normalize_dataset(l1, l2, model.tokenizer)
            original_sentences_list.extend(original_sentences)
            original_labels_list.extend(original_labels)

            # Remove tags
            text = text.replace('<d>', '')
            text = text.replace('</d>', '')
            text = text.replace('<C>', '')
            text = text.replace('</C>', '')
            text = text.replace('<M>', '')
            text = text.replace('</M>', '')

            # # Add \n after "。" which do not already have it
            # text = re.sub('。(?=[^\n])', "。\n", text)
            #
            # # Apply the model to extract symptoms
            # sentences = text.split('\n')
            sentences = list()
            sentences.append(text)
            sentences_embeddings = [model.tokenizer.convert_tokens_to_ids(['[CLS]'] + t) for t in original_sentences]
            labels = model.predict(sentences_embeddings)
            sentences = model.convert_ids_to_tokens(sentences_embeddings)
            predict_labels_list.extend(labels)

            tagged_sentences = list()
            for sent, label in zip(sentences, labels):
                tagged_sentences.append(iob_util.convert_iob_to_xml(sent, label))
            output_file.write("Text " + str(text_num) + "\n\n")
            output_file.write("\n".join(tagged_sentences))
            output_file.write("\n\n\n")

        print('')
        output_file.close()

    iob_util.evaluate_performance(original_labels_list, predict_labels_list)
