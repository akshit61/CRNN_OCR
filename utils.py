import torch
from sklearn import preprocessing
import numpy as np
alphabets = '0123456789,.:(%$!^&-/);<~|`>?+=_[]{}"\'@#*ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\ '


class strLabelConverter(object):
    def __init__(self, alphabets):
        self.alphabets = alphabets
        self.dict_str2num = {}
        self.dict_num2str = {}
        count = len(alphabets)
        for i, char in enumerate(alphabets):
            self.dict_str2num[char] = i + 1
        self.dict_str2num['<u>'] = count+1
        for k,v in self.dict_str2num.items():
            self.dict_num2str[v] = k
        self.dict_num2str[0] = '°'

    def encode(self, text):
        length = []
        result = []
        for item in text:
            item = item
            length.append(len(item))
            r = []
            for c in item:
                if c in self.dict_str2num:
                    r.append(self.dict_str2num[c])
                else:
                    r.append(self.dict_str2num['<u>'])
            result.append(r)

        max_length = max(length)
        result_temp = []
        for item in result:
            for _ in range(max_length - len(item)):
                item.append(0)
            result_temp.append(item)
        text = result_temp
        return torch.LongTensor(text), torch.LongTensor(length)

    def decode(self, t, length, raw=False):

        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                    length)
            if raw:
                return ''.join([self.dict_num2str[i.item()] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.dict_num2str[t[i].item()])
                    #char_list.append(self.dict_num2str[t[i].item()])
                return ''.join(char_list)
        else:
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            index = 0
            texts = []
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


class averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.numel()
        v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
