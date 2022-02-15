import pandas as pd
from lxml.etree import XMLSyntaxError

from BERT.predict import load_model, predict
from util import iob_util
from BERT.util.bert_utils import normalize_dataset
from util.iob_util import convert_xml_to_iob
from util.xml_parser import __preprocessing, split_sentences, drop_texts_with_mismatched_tags
from seqeval.metrics import accuracy_score, f1_score, precision_score, classification_report
from seqeval.scheme import IOB2

from dnorm_j import DNorm


def list_size(list):
    return sum([len(t) for t in list])

def flatten_list(list):
    flat_list = [item for sublist in list for item in sublist]
    return flat_list

def convert_text_to_iob_list(texts, tag_list, ignore_mismatch_tags=True):
    # Preprocess
    texts = __preprocessing(texts)

    # Convert
    items = list()
    tags = list()
    i = 0
    for t in texts:
        sent = list()
        tag = list()
        try:
            iob = convert_xml_to_iob(t, tag_list, ignore_mismatch_tags=ignore_mismatch_tags)
            # Convert tuples into lists
            for item in iob:
                if item[0] == ' ':
                    continue
                sent.append(item[0])
                tag.append(item[1])
            items.append(sent)
            tags.append(tag)
            i = i + 1
        except XMLSyntaxError:
            print("Skipping text with xml syntax error, id: " + str(i))
    return items, tags


if __name__ == '__main__':
    ##### Load model #####
    MODEL = 'cl-tohoku/bert-base-japanese-char-v2'
    model, tokenizer, id2label = load_model(MODEL, '../BERT/out')
    TAG_LIST = ['C']

    #### Load data #####
    file = '../data/DATA_IM_v6.txt'
    data = pd.read_csv(file, sep="	")
    texts_tagged = data['text_tagged'].tolist()
    texts_raw = data['text_raw'].tolist()

    texts = __preprocessing(texts_raw)
    texts = split_sentences(texts_raw)
    texts = drop_texts_with_mismatched_tags(texts_raw)

    # Get iob info from xml as ground true labels
    original_sentences, original_labels = convert_text_to_iob_list(texts_tagged, TAG_LIST)

    ##### Tokenize text for BERT #####
    # print(sum([len(t) for t in texts]))
    texts = [tokenizer.tokenize(t) for t in texts]
    data_x = [tokenizer.convert_tokens_to_ids(['[CLS]'] + t) for t in texts]

    # Normalize to same tokenization as BERT
    original_sentences, original_labels = normalize_dataset(original_sentences, original_labels, tokenizer)

    ##### Extract drug names #####
    tags = predict(model, data_x)
    labels = [[id2label[t] for t in tag] for tag in tags]
    data_x = [tokenizer.convert_ids_to_tokens(t)[1:] for t in data_x]

    # Remove pad tags from labels
    labels = [sent_label[:len(sent)] for sent, sent_label in zip(data_x, labels)]

    ##### Insanity check #####
    assert list_size(original_sentences) == list_size(data_x) == list_size(original_labels) == list_size(labels)

    ##### Save output iob file #####
    correct = 0
    f = open("out/iob_predict_" + file.split('/')[-1] + ".iob", 'w')
    for original_sentence, original_sentence_label, output_sentence, predict_sentence_label in zip(original_sentences, original_labels, data_x, labels):
        for original_char, original_char_label, output_char, predict_char_label in zip(original_sentence, original_sentence_label, output_sentence, predict_sentence_label):
            line = original_char + '\t' + original_char_label + '\t' + output_char + '\t' + predict_char_label + '\n'
            f.write(line)
            if original_char_label == predict_char_label:
                correct = correct + 1
        f.write('\n')
    f.close()

    ###### Calculate perfromance metrics #####
    print('Accuracy: ' + str(accuracy_score(original_labels, labels)))
    print('Precision: ' + str(precision_score(original_labels, labels)))
    print('F1 score: ' + str(f1_score(original_labels, labels)))
    #print(classification_report(original_labels, labels))
    print(classification_report(original_labels, labels, mode='strict', scheme=IOB2))

    output = pd.DataFrame()
    i = 0

    ##### Match tags to UMLS ####
    for sent_number in range(len(original_sentences)):

        ne_dict = iob_util.convert_iob_to_dict(original_sentences[sent_number], labels[sent_number])

        # Normalize
        normalized_entities = list()
        normalization_model = DNorm.from_pretrained()
        for entry in ne_dict:
            named_entity = entry['word']
            normalized_named_entity = normalization_model.normalize(named_entity)
            normalized_entities.append(normalized_named_entity)
        df = pd.DataFrame(ne_dict)
        df['normalized'] = normalized_entities

        # Search on UMLS
        import umls_api

        cuis = list()
        for entity in normalized_entities:
            results = umls_api.API(api_key='29d4de15-33b3-465c-bd93-f71c3712d55e').term_search(entity)
            try:
                i = i + 1
                cui = results['result']['results'][0]['ui']
                print(i, cui)
            except Exception:
                cui = 0
            cuis.append(cui)
        df['cui'] = cuis
        df.insert(0, 'Sentence', sent_number)
        output = output.append(df)
