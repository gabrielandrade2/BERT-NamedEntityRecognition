import json
import os
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch import nn, optim
from tqdm import tqdm
from transformers import BertForTokenClassification, BertJapaneseTokenizer
from transformers import get_linear_schedule_with_warmup

from BERT.train import train
from BERT.util import data_utils, bert_utils
from util.xml_parser import convert_xml_to_iob_list


def train_from_xml_file(xmlFile, model, tag_list, output_dir):
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print("folder exists")

    ##### Load the data #####
    sentences, tags = convert_xml_to_iob_list(xmlFile, tag_list, should_split_sentences=True)
    return train_from_sentences_tags_list(sentences, tags, model, output_dir)


def train_from_sentences_tags_list(sentences, tags, model, output_dir):
    ##### Process dataset for BERT #####
    tokenizer = BertJapaneseTokenizer.from_pretrained(model)

    # Create vocabulary
    label_vocab = bert_utils.create_label_vocab(tags)
    with open(output_dir + '/label_vocab.json', 'w') as f:
        json.dump(label_vocab, f, ensure_ascii=False)

    ##### Split in train/validation #####
    x_train, x_test, y_validation, y_validation = train_test_split(sentences, tags, test_size=0.2)

    # Convert to BERT data model
    input_x, input_y = bert_utils.dataset_to_bert_input(x_train, y_validation, tokenizer, label_vocab)
    val_x, val_y = bert_utils.dataset_to_bert_input(x_test, y_validation, tokenizer, label_vocab)

    # Get pre-trained model and fine-tune it
    model = BertForTokenClassification.from_pretrained(model, num_labels=len(label_vocab))
    model = train(model, input_x, input_y, val=[val_x, val_y], outputdir=output_dir)

    #print(model)
    return model


def train(model, x, y, max_epoch=10, lr=3e-5, batch_size=8, val=None, outputdir=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ' + device)
    data = data_utils.Batch(x, y, batch_size=batch_size)
    if val is not None:
        val_data = data_utils.Batch(val[0], val[1], batch_size=batch_size)
        val_loss = []

    loss = nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_step = int((len(data)//batch_size)*max_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_step*0.1), total_step)

    losses = []
    min_val_loss = 999999999999
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
        print(losses)

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
        print(min_epoch)
        model_path = outputdir + '/checkpoint{}.model'.format(min_epoch)
        model.load_state_dict(torch.load(model_path))

    torch.save(model.state_dict(), outputdir+'/final.model')
    return model


if __name__ == '__main__':

    xmlFile = '../data/drugHistoryCheck.xml'
    model = 'cl-tohoku/bert-base-japanese-char-v2'
    tag_list = ['d']
    output_dir = 'out'

    train_from_xml_file(xmlFile, model, tag_list, output_dir)