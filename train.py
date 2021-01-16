import os
import glob
import torch
import numpy as np
import pandas as pd
import pickle

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import utils
import matplotlib.pyplot as plt
from engine import train_fn, eval_fn
import model

# alphabets = '0123456789,.:(%$!^&-/);<~|`>?+=_[]{}"\'@#*ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\ '

converter = utils.strLabelConverter(config.ALPHABETS)
crnn = model.CRNN(config.IMG_HEIGHT, 3)
crnn.to(config.DEVICE)


if config.PRETRAINED:
    print('loading pretrained model')
    crnn.load_state_dict(torch.load(config.PRETRAINED))

optimizer = torch.optim.Adam(crnn.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)


def read_pickle(dataset):
    with open(os.path.join(config.DATA_ROOT, dataset + '.pickle'), 'rb') as handle:
        data = pickle.load(handle)
    image_files, targets_orig = [], []
    for k, v in data.items():
        image_files.append(os.path.join(config.DATA_ROOT, dataset, k + '.jpg'))
        targets_orig.append(v)

    return image_files, targets_orig


def run_training():
    img_files, target_orig = read_pickle('train')
    train_dataset = dataset.OCR_data(img_files, target_orig, resize=(config.IMG_HEIGHT, config.IMG_WIDTH))

    img_files, target_orig = read_pickle('valid')
    test_dataset = dataset.OCR_data(img_files, target_orig, resize=(config.IMG_HEIGHT, config.IMG_WIDTH))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                               num_workers=config.NUM_WORKERS, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                                              num_workers=config.NUM_WORKERS, shuffle=True)

    for epoch in range(config.EPOCHS):
        train_loss = train_fn(crnn, train_loader, optimizer)
        print('Epoch {0} : Train Loss = {1}'.format(epoch, train_loss))
        eval_fn(crnn, test_loader)
        if epoch%(config.SAVE_INTERVAL) == 0:
            torch.save(crnn.state_dict(), '{0}/crnn_epoch_{1}.pth'.format(config.EXPR_DIR, epoch))

if __name__ == '__main__':
    run_training()
