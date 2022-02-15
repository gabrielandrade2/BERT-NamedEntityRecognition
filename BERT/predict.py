import torch
import json
import numpy as np
from transformers import BertJapaneseTokenizer, BertForTokenClassification
from BERT.util import data_utils, bert_utils


def load_model(model, model_dir):
    tokenizer = BertJapaneseTokenizer.from_pretrained(model)

    with open(model_dir + '/label_vocab.json', 'r') as f:
        label_vocab = json.load(f)
    id2label = {v: k for k, v in label_vocab.items()}

    model = BertForTokenClassification.from_pretrained(model, num_labels=len(label_vocab))
    model_path = model_dir + '/final.model'
    model.load_state_dict(torch.load(model_path))

    return model, tokenizer, id2label

def predict_from_sentences_list(sentences, model, tokenizer, vocabulary, device):
    sentences_embeddings = bert_utils.prepare_sentences(sentences, tokenizer)
    tags = predict(model, sentences_embeddings, device=device)
    labels = convert_prediction_to_labels(tags, vocabulary)
    sentences = [tokenizer.convert_ids_to_tokens(t)[1:] for t in sentences_embeddings]
    labels = remove_label_padding(sentences, labels)
    return sentences, labels

def predict(model, x, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device: ' + device)

    data = data_utils.Batch(x, x, batch_size=8, sort=False)

    model.to(device)
    model.eval()

    res = []

    for sent, _, _ in data:
        sent = torch.tensor(sent).to(device)
        mask = [[float(i > 0) for i in ii] for ii in sent]
        mask = torch.tensor(mask).to(device)

        output = model(sent, attention_mask=mask)
        logits = output[0].detach().cpu().numpy()
        tags = np.argmax(logits, axis=2)[:, 1:].tolist()
        res.extend(tags)

    return res


def convert_prediction_to_labels(prediction, vocabulary):
    return [[vocabulary[t] for t in tag] for tag in prediction]


def remove_label_padding(sentences, labels):
    new_labels = list()
    for sent, label in zip(sentences, labels):
        new_labels.append(label[:len(sent)])
    return new_labels
