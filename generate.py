# import some packages you need here
import torch
import numpy as np 
import pandas as pd 
import string 
from model import CharRNN, CharLSTM
import torch.nn.functional as F

def generate(model, seed_characters, temperature, device, length = 100, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """
    model.eval()
    hidden = model.init_hidden(1)
    result = []

    data = open(r'./shakespeare_train.txt').read().strip()
    char = list(sorted(set(data)))
    char_to_ix = {ch: i for i, ch in enumerate(char)}

    for i in range(length) :
        if i == 0:
            start = torch.tensor([char_to_ix[c] for c in seed_characters], dtype=torch.int64).cuda(device)
            x0 = start.unsqueeze(0)
            output,hidden = model(x0)
            out_dist = F.softmax(output.squeeze()/temperature)
            top_i = torch.multinomial(out_dist, 1)[0]
            result.append(top_i)
        else :
            inp = torch.tensor([[top_i]], dtype=torch.int64)
            inp = inp.cuda(device)
            output, hidden = model(inp,hidden)
            
            out_dist = F.softmax(output.squeeze()/temperature)
            top_i = torch.multinomial(out_dist,1)[0]
            result.append(top_i)

    return seed_characters + ''.join(char[i] for i in result)

if __name__ == '__main__':
    device = 0
    types = 'LSTM'
    pre = torch.load(f"./checkpoint_h/{types}/best_model.pth")
    model = CharLSTM(hidden_size = 512, device = device).cuda(device)
    model.load_state_dict(pre)
    for w in ['A', 'I', 'F', 'W', 'M'] : 
        result = generate(model,w, 100, device)
        print('                                                                                                 ')
        print(f'=============================================={w}=================================================================')
        print(result)
        print('==================================================================================================================')
        print('                                                                                                 ')
