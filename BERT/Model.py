import json
import os

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, BertJapaneseTokenizer, BertForTokenClassification

from BERT import data_utils


class NERModel:
    def __init__(self, pre_trained_model, tokenizer, vocabulary, device='cpu'):
        self.model = pre_trained_model
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.device = device

    @classmethod
    def load_transformers_model(cls, pre_trained_model_name, model_dir, device='cpu'):
        tokenizer = BertJapaneseTokenizer.from_pretrained(pre_trained_model_name)

        with open(model_dir + '/label_vocab.json', 'r') as f:
            label_vocab = json.load(f)

        model = BertForTokenClassification.from_pretrained(pre_trained_model_name, num_labels=len(label_vocab))
        model_path = model_dir + '/final.model'
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))

        return cls(model, tokenizer, label_vocab, device)

    def train(self, x, y, max_epoch=10, lr=3e-5, batch_size=8, val=None, outputdir=None):
        model = self.model
        device = self.device

        os.makedirs(outputdir, exist_ok=True)

        data = data_utils.Batch(x, y, batch_size=batch_size)
        if val is not None:
            val_data = data_utils.Batch(val[0], val[1], batch_size=batch_size)
            val_loss = []

        optimizer = optim.Adam(model.parameters(), lr=lr)
        total_step = int((len(data) // batch_size) * max_epoch)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_step * 0.1), total_step)

        losses = []
        model.to(device)
        for epoch in tqdm(range(max_epoch)):
            print('EPOCH :', epoch+1)
            model.train()
            all_loss = 0
            step = 0

            for sent, label, _ in data:
                sent = torch.tensor(sent).to(device)
                label = torch.tensor(label).to(device)
                mask = [[float(i>0) for i in ii] for ii in sent]
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
                    mask = [[float(i>0) for i in ii] for ii in sent]
                    mask = torch.tensor(mask).to(device)

                    output = model(sent, attention_mask=mask, labels=label)
                    loss = output[0]
                    all_loss += loss.item()

                    step += 1
                val_loss.append(all_loss / step)
                output_path = outputdir + '/checkpoint{}.model'.format(len(val_loss)-1)
                torch.save(model.state_dict(), output_path)

        if val is not None:
            min_epoch = np.argmin(val_loss)
            #print(min_epoch)
            model_path = outputdir + '/checkpoint{}.model'.format(min_epoch)
            model.load_state_dict(torch.load(model_path))

        torch.save(model.state_dict(), outputdir+'/final.model')
        self.model = model
        return model

    def predict(self, x):
        model = self.model
        device = self.device

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

        return res
