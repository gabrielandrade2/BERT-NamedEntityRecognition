import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from util.bert import data_utils


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
