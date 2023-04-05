import argparse
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from net import *
import numpy as np

from util import *
from trainer import Optim
import datetime
 
def fn_metric(pred, y):
	rse = np.sqrt(np.mean((y - pred)**2))/ np.sqrt(np.mean((y-y.mean())**2))
	rae = np.mean(abs(y - pred))/ np.mean(abs(y-y.mean()))
	return rse, rae

global_log_file = None
def pprint(*args):
    # print with UTC-8 time
    time = '['+str(datetime.datetime.now())[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)

def get_model(args):

    if args.model_name == 'GRU':
        return GRUModel(d_feat=args.d_feat, hidden_size=args.hidden_size, num_layers=args.num_gru_layer)
    
    if args.model_name == 'Causal':
        return CausalModel(d_feat=args.d_feat, hidden_size=args.hidden_size, num_layers=args.num_gru_layer, base_model=args.base_model)
    
    if args.model_name == 'Causal_ATT':
        return CausalModel_ATT(d_feat=args.d_feat, hidden_size=args.hidden_size, num_layers=args.num_gru_layer, base_model=args.base_model)
    
    raise ValueError('unknown model name `%s`'%args.model_name)


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    length = len(X) // batch_size
    for x, y in tqdm(data.get_batches(X, Y, batch_size, True), total=length):
        model.zero_grad()
        x = x.transpose(1,2)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)
        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id, dtype=torch.long).to(device)
            tx = x[:, id, :]
            ty = y[:, id]
            output = model(tx)
            # output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:,id]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        iter += 1
    return total_loss / n_samples

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = []
    test = []
    length = len(X) // batch_size
    for X, Y in tqdm(data.get_batches(X, Y, batch_size, False), total=length):
        X = X.transpose(1,2)
        with torch.no_grad():
            output = model(X)
        test.append(Y)

        predict.append(output)

        # output= torch.squeeze(output)
        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)
    predict = torch.cat(predict)
    test = torch.cat(test)


    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation, predict, test

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--device',type=str,default='cuda:4',help='')
parser.add_argument('--data', type=str, default='./data/electricity.txt',
                    help='location of the data file')
parser.add_argument('--model_name', type=str, default='Causal',
                    help='model')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=int, default=1)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes/variables')
parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)

parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--early_stop', type=int, default=100)
parser.add_argument('--lradj',type=int,default=1,help='lradj')

parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')
parser.add_argument('--d_feat', type=int, default=7)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_gru_layer', type=int, default=1)
parser.add_argument('--k_day', type=int, default=10)
parser.add_argument('--n_neighbor', type=int, default=10)
parser.add_argument('--hidden_batch_size', type=int, default=128)
parser.add_argument('--base_model', type=str, default='GRU',
                    help='base model of causal model')


args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

def main():
    global global_log_file
    global_log_file = args.save + '.' + 'run.log'
    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)
    model = get_model(args)
    model = model.to(device)
    pprint(args)
    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    pprint('Number of model parameters is', nParams)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    best_val = 1000000
    stop_round = 0
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(args.epochs ):
            epoch_start_time = time.time()
            # lr = adjust_learning_rate(optim.optimizer, epoch, args)
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)

            val_loss, val_rae, val_corr, _, _ = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                               args.batch_size)
            pprint(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
            # test_acc, test_rae, test_corr, _, _ = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
            #                                          args.batch_size)
            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
                stop_round = 0
                test_acc, test_rae, test_corr, _, _ = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                     args.batch_size)
                pprint("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
            else:
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    vtest_acc, vtest_rae, vtest_corr, _, _ = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    test_acc, test_rae, test_corr, test_predict, test_test = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    
    pprint("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
    return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr

if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    for i in range(1):
        val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main()
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
    pprint('\n\n')
    pprint('1 runs average')
    pprint('\n\n')
    pprint("valid\trse\trae\tcorr")
    pprint("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    pprint("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    pprint('\n\n')
    pprint("test\trse\trae\tcorr")
    pprint("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
    pprint("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))

