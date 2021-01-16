from tqdm import tqdm
import torch
import config
from torch import nn
import utils
import model
from torch.nn import functional as f
from torch.autograd import Variable


criterion = nn.CTCLoss(blank=0, zero_infinity=True)
converter = utils.strLabelConverter(config.ALPHABETS)
crnn = model.CRNN(config.IMG_HEIGHT, nc=3)


image = torch.FloatTensor(config.BATCH_SIZE, 3, config.IMG_HEIGHT, config.IMG_WIDTH)
text = torch.LongTensor(config.BATCH_SIZE * 5)
length = torch.LongTensor(config.BATCH_SIZE)

if config.DEVICE == 'cuda' and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)


def train_fn(model, data_loader, optimizer):
    model.train()
    tk = tqdm(data_loader, total=len(data_loader))
    fin_loss = 0
    loss_avg = utils.averager()
    for data in tk:
        imgs, texts = data.values()
        utils.loadData(image, imgs)
        batch_size = imgs.size(0)
        t, l = converter.encode(texts)
        utils.loadData(text, t)
        utils.loadData(length, l)
        optimizer.zero_grad()
        preds = model(image)
        preds_length = torch.full(size=(batch_size,), fill_value=preds.size(0), dtype=torch.int32)
        loss = criterion(preds, text, preds_length, length)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    tk = tqdm(data_loader, total=len(data_loader))
    n_correct = 0
    loss_avg = utils.averager()
    with torch.no_grad():
        for data in tk:
            imgs, texts = data.values()
            utils.loadData(image, imgs)
            batch_size = imgs.size(0)
            t, l = converter.encode(texts)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = model(image)
            # print(preds.size())
            preds_length = torch.full(size=(batch_size,), fill_value=preds.size(0), dtype=torch.int32)
            loss = criterion(preds, text, preds_length, length)
            loss_avg.add(loss)
            preds = f.softmax(preds,dim=2)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds, preds_length)
            cpu_texts_decode = []
            for i in texts:
                cpu_texts_decode.append(i)
            for pred, target in zip(sim_preds, cpu_texts_decode):
                if pred == target:
                    n_correct += 1

        raw_preds = converter.decode(preds, preds_length, raw=True)[:10]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

        accuracy = n_correct / float(len(data_loader) * config.BATCH_SIZE)
        print('Val.loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))
