import json
import os

import mojimoji
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, BertJapaneseTokenizer, BertForTokenClassification

from BERT import data_utils
from util import text_utils

CLS_TAG = '[CLS]'
PAD_TAG = '[PAD]'


class NERModel:
    def __init__(self, pre_trained_model, tokenizer, vocabulary, device='cpu'):
        self.model = pre_trained_model
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.device = device
        self.max_size = None

    @classmethod
    def load_transformers_model(cls, pre_trained_model_name, model_dir, device='cpu'):
        tokenizer = BertJapaneseTokenizer.from_pretrained(pre_trained_model_name)

        with open(model_dir + '/label_vocab.json', 'r') as f:
            label_vocab = json.load(f)

        model = BertForTokenClassification.from_pretrained(pre_trained_model_name, num_labels=len(label_vocab))
        model_path = model_dir + '/final.model'
        model.load_state_dict(torch.load(model_path, map_location=device))

        return cls(model, tokenizer, label_vocab, device)

    def set_max_size(self, max_size):
        self.max_size = max_size

    def train(self, x, y, max_epoch=10, lr=3e-5, batch_size=8, val=None, outputdir=None):
        model = self.model
        device = self.device
        max_size = self.max_size

        if not max_size:
            max_size = max([len(i) for i in x])

        os.makedirs(outputdir, exist_ok=True)

        data = data_utils.Batch(x, y, batch_size=batch_size, max_size=max_size)
        if val is not None:
            val_data = data_utils.Batch(val[0], val[1], batch_size=batch_size)
            val_loss = []

        optimizer = optim.Adam(model.parameters(), lr=lr)
        total_step = int((len(data) // batch_size) * max_epoch)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_step * 0.1), total_step)

        losses = []
        model.to(device)
        for epoch in tqdm(range(max_epoch)):
            print('EPOCH :', epoch + 1)
            model.train()
            all_loss = 0
            step = 0

            for sent, label, _ in data:
                sent = torch.tensor(sent).to(device)
                label = torch.tensor(label).to(device)
                mask = [[float(i > 0) for i in ii] for ii in sent]
                mask = torch.tensor(mask).to(device)

                output = model(sent, attention_mask=mask, labels=label)
                loss = output[0]
                all_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                step += 1

            losses.append(all_loss / step)

            if val is not None:
                model.eval()
                all_loss = 0
                step = 0

                for sent, label, _ in val_data:
                    sent = torch.tensor(sent).to(device)
                    label = torch.tensor(label).to(device)
                    mask = [[float(i > 0) for i in ii] for ii in sent]
                    mask = torch.tensor(mask).to(device)

                    output = model(sent, attention_mask=mask, labels=label)
                    loss = output[0]
                    all_loss += loss.item()

                    step += 1
                val_loss.append(all_loss / step)
                output_path = outputdir + '/checkpoint{}.model'.format(len(val_loss) - 1)
                torch.save(model.state_dict(), output_path)

        if val is not None:
            min_epoch = np.argmin(val_loss)
            # print(min_epoch)
            model_path = outputdir + '/checkpoint{}.model'.format(min_epoch)
            model.load_state_dict(torch.load(model_path))

        torch.save(model.state_dict(), outputdir + '/final.model')
        self.model = model
        return model

    def predict(self, x, return_labels=True):
        model = self.model
        device = self.device
        max_size = self.max_size

        if not max_size:
            max_size = max([len(i) for i in x])

        data = data_utils.Batch(x, x, batch_size=8, sort=False, max_size=max_size)

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

        res = self.__remove_label_padding(x, res)

        if return_labels:
            return self.convert_prediction_to_labels(res)
        else:
            return res

    def prepare_sentences(self, sentences, split_sentences=False):
        """ Prepare sentences from model execution (tokenization + CLS tag addition).

        :param sentences: The list of sentences to be prepared.
        :param split_sentences: Should texts be split into sentences?
        :return: The list of prepared sentences.
        """
        if isinstance(sentences, str):
            temp = list()
            temp.append(sentences)
            sentences = temp

        if split_sentences:
            single_item = len(sentences) == 1
            sentences = text_utils.split_sentences(sentences, single_item)

        sentences = [mojimoji.han_to_zen(sentence) for sentence in sentences]
        tokenized_sentences = [self.tokenizer.tokenize(t) for t in sentences]
        return [self.tokenizer.convert_tokens_to_ids([CLS_TAG] + t) for t in tokenized_sentences]

    def normalize_tagged_dataset(self, sentences, tags):
        """ Use the tokenizer to apply the same normalization applied to full strings to a pre-tokenized dataset. It also
        adjusts the referring tags in case of removal/expansion of tokens.

        :param sentences: A list of list of character tokenized sentences.
        :param tags: A list of list of tags corresponding to the sentences.
        :return: Two lists, the processed sentences and the adjusted labels.
        """

        processed_sentences = list()
        processed_tags_sentences = list()

        for sentence, tag_sentence in zip(sentences, tags):
            processed_sentence = list()
            processed_tag_sentence = list()
            for character, tag_character in zip(sentence, tag_sentence):
                tokenized = self.tokenizer.tokenize(character)
                last_tag = str()
                for token in tokenized:
                    if token == '' or token == ' ':
                        continue
                    processed_sentence.append(token)

                    # In the case we are expanding a character that is tagged with a Beggining tag, we make the subsequent
                    # ones as Intra.
                    if last_tag.startswith('B') and last_tag == tag_character:
                        tag_character = tag_character.replace('B', 'I', 1)
                    processed_tag_sentence.append(tag_character)

            processed_sentences.append(processed_sentence)
            processed_tags_sentences.append(processed_tag_sentence)
        return processed_sentences, processed_tags_sentences

    def convert_prediction_to_labels(self, prediction):
        id2label = {v: k for k, v in self.vocabulary.items()}
        return [[id2label[t] for t in tag] for tag in prediction]

    @staticmethod
    def __remove_label_padding(sentences, labels):
        new_labels = list()
        for sent, label in zip(sentences, labels):
            new_labels.append(label[:len(sent) - 1])
        return new_labels

    def convert_ids_to_tokens(self, embeddings):
        temp = [self.tokenizer.convert_ids_to_tokens(t)[1:] for t in embeddings]
        return [[mojimoji.han_to_zen(t) for t in sent] for sent in temp]
