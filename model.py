import torch.nn as nn
import torch
from torch.autograd import Variable

class CharRNN(nn.Module):

    def __init__(self, input_size = 62, hidden_size = 100, n_layers=3, device = 0):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_layer = nn.RNN(hidden_size, hidden_size, n_layers,batch_first = True)
        self.decode = nn.Linear(hidden_size, input_size)
        self.device = device

    def init_hidden(self, batch_size):

        initial_hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda(self.device)
        
        return initial_hidden

    def forward(self, input, hidden = None):
        
        x = self.embedding(input)
        output, hidden = self.rnn_layer(x, hidden)
        output = self.decode(output)

        return output, hidden

class CharLSTM(nn.Module):
    def __init__(self, input_size = 62, hidden_size = 100, n_layers=3, device = 0):
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm_layer = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first = True)
        self.decode = nn.Linear(hidden_size, input_size)
        self.device = device

    def init_hidden(self, batch_size):

        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda(self.device)
        cell = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda(self.device)
        
        return (hidden,cell)

    def forward(self, input, hidden = None):

        
        x = self.embedding(input)
        output, hidden = self.lstm_layer(x, hidden)
        output = self.decode(output)

        return output, hidden

if __name__ == '__main__':
    import dataset
    ds = dataset.Shakespeare("./shakespeare_train.txt", chuck_size=30)
    loader = da.DataLoader(ds, batch_size=1, shuffle=False)
    all_chars = string.printable
    aa = CharRNN(100, 100, 100, 3)
    hidden = aa.init_hidden(1)
    for i, j in loader :
        aa(i, hidden)