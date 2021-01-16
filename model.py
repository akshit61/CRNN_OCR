import torch
from torch import nn
from torch.nn import functional as f


class CRNN(nn.Module):
    def __init__(self, imgH, nc):
        super(CRNN, self).__init__()
        ks = [3, 3, 3, 3, 3, 3]
        ss = [1, 1, 1, 1, 1, 1]
        ps = [1, 1, 1, 1, 1, 1]
        nm = [32, 64, 128, 256, 256, 512]
        assert imgH % 16 == 0
        cnn = nn.Sequential()

        def cnnLayer(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module(f'BatchNorm{i}', nn.BatchNorm2d(nOut))
            cnn.add_module(f'relu{i}', nn.ReLU(True))

        cnnLayer(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        cnnLayer(1, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        cnnLayer(2)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))
        cnnLayer(3, True)
        cnnLayer(4)
        cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        cnnLayer(5)
        cnn.add_module('pooling{0}'.format(5), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        self.cnn = cnn
        self.rnn = rnnLayer(512, 64, 2)
        self.fc = nn.Linear(128, 97)

    def forward(self, x):
        out = self.cnn(x)
        out = out.squeeze(2)
        out = out.permute(2, 0, 1)
        out = self.rnn(out)
        t, b, h = out.size()
        out = out.view(t * b, h)
        out = self.fc(out)
        out = out.view(t, b, -1)
        out = f.log_softmax(out,dim = 2)
        #print(out.size())
        return out

class rnnLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(rnnLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


if __name__ == '__main__':
    model = CRNN(32, 3)
    x = torch.rand(1, 3, 32, 200)
    model(x)
    '''model = rnnLayer(512,64,2)
    x = torch.randn(27,1,512)
    model(x)'''
