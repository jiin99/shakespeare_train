import dataset
import numpy as np
from model import CharRNN, CharLSTM
import torch
import time
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
import matplotlib.pyplot as plt 
import seaborn as sns 
import os


def train(model, trn_loader, device, criterion, optimizer, epoch):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """
    trn_loss = 0
    model.train()
    for i, (data,target) in enumerate(trn_loader):
        hidden = model.init_hidden(data.shape[0])
        start_time = time.time()

        data, target = data.cuda(device), target.cuda(device)

        output,hidden = model(data,hidden)

        optimizer.zero_grad()

        loss = criterion(output.view(-1,62),target.view(-1,))

        loss.backward()
        optimizer.step()
        trn_loss += loss.item()

        end_time = time.time()
        if i%10 == 0:
            print(" [Train][{0}] [{1}/{2}] Losses : [{3:.4f}] Time : [{4:.2f}]".format(epoch, i, len(trn_loader), loss.item(), end_time-start_time))
    
    trn_loss = trn_loss/len(trn_loader)

    return trn_loss

def validate(model, val_loader, device, criterion, epoch):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """
    val_loss = 0
    
    model.eval()
    with torch.no_grad():
        for i,(data,target) in enumerate(val_loader):
            hidden = model.init_hidden(data.shape[0])
            start_time = time.time()
            data, target = data.to(device), target.to(device)

            output,hidden = model(data,hidden)
            loss = criterion(output.view(-1,62),target.view(-1,))

            val_loss += loss.item()
            end_time = time.time()
            if i%10 == 0:
                print("[{0}] [{1}/{2}] Losses : [{3:.4f}] Time : [{4:.2f}]".format(epoch, i, len(val_loader), loss.item(), end_time-start_time))

    
    val_loss = val_loss/len(val_loader)

    return val_loss


def main(types = 'RNN'):
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    device = 0
    epochs = 50

    criterion = nn.CrossEntropyLoss()

    dl = dataset.Shakespeare("./shakespeare_train.txt")
    
    total_idx = list(range(len(dl)))
    np.random.shuffle(total_idx)
    split_idx = int(len(dl)*0.7)
    trn_idx = total_idx[:split_idx]
    val_idx = total_idx[split_idx:]

    trn_loader = data.DataLoader(dl, batch_size = 1024, sampler = SubsetRandomSampler(trn_idx))
    val_loader = data.DataLoader(dl, batch_size = 1024, sampler = SubsetRandomSampler(val_idx))

    if types == 'RNN' : 
        model = CharRNN(hidden_size = 512).cuda(device)
    elif types == 'LSTM':
        model = CharLSTM(hidden_size = 512).cuda(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    val_losses = []

    best_valid = 5.0
    for epoch in range(1,1+epochs) :
        loss = train(model, trn_loader, device, criterion, optimizer, epoch)
        val_loss = validate(model, val_loader, device, criterion, epoch)
        losses.append(loss)
        val_losses.append(val_loss)

        if val_loss < best_valid :
            best_valid = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),os.path.join(f'./checkpoint_h/{types}','best_model.pth'))

    return losses, val_losses, best_epoch

def draw_curve(logger_train, logger_test, best_epoch, types='RNN') :

    fig = plt.gcf()
    plt.plot(logger_train, c='blue', label = "Training Loss")
    plt.plot(logger_test, c='red', label = "Validation Loss")
    plt.title("{0} Loss , best epoch : {1}".format(types,best_epoch))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(f'./checkpoint_h/{types}/Loss.png')
    plt.close()
    
    from pandas import DataFrame
    tr_df = DataFrame(logger_train, columns = ['train_loss'])
    val_df = DataFrame(logger_test, columns = ['val_loss'])

    tr_df.to_csv(f'./checkpoint_h/{types}/train_log.csv')
    val_df.to_csv(f'./checkpoint_h/{types}/val_log.csv')

if __name__ == '__main__':
    trn_loss, val_loss, best_epoch = main(types = 'RNN')
    draw_curve(trn_loss, val_loss, best_epoch, types = 'RNN')
    # trn_loss, val_loss , best_epoch = main(types = 'LSTM')
    # draw_curve(trn_loss, val_loss, best_epoch ,types = 'LSTM')