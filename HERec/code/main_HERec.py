import numpy
import torch
from utils import load_data, set_params, evaluate
from dgl.nn.pytorch import DeepWalk
from torch.utils.data import DataLoader
from torch.optim import SparseAdam
import warnings
import datetime
import pickle as pkl
import scipy.sparse as sp
from embedding_generation import sparse_mx_to_torch_sparse_tensor, normalize_adj, tensorToDGL
import random

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train():
    mps, label, idx_train, idx_val, idx_test = load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    # feats_dim_list = [i.shape[1] for i in feats]
    gs = tensorToDGL(mps)
    g = gs[1]
    print("seed ", args.seed)
    print("Dataset: ", args.dataset)
    print("The meta-paths: PRP")

    model = DeepWalk(g, walk_length=5, window_size=3).to(device)
    optimiser = SparseAdam(model.parameters(), lr=args.lr)
    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0
    average_loss = 0.

    starttime = datetime.datetime.now()
    dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=128,
                            shuffle=True, collate_fn=model.sample)
    for epoch in range(args.nb_epochs):
        for batch_walk in dataloader:
            batch_walk = batch_walk.to(device)
            model.train()
            loss = model(batch_walk)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            average_loss += loss.item()
        average_loss = average_loss / len(dataloader)
        print("epoch {:d} | loss {:.4f} ".format(epoch, average_loss))
        if average_loss < best:
            best = average_loss
            best_t = epoch
            cnt_wait = 0
            embeds = model.node_embed.weight.detach()
            torch.save(embeds, 'HERec_' + own_str + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    print('Loading {}th epoch'.format(best_t))
    embeds = torch.load('HERec_' + own_str + '.pkl')
    embeds.requires_grad = False

    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label,
                 nb_classes, device, args.dataset, args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")

    if args.save_emb:
        f = open("./embeds/" + args.dataset + "/" + str(args.turn) + ".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()


if __name__ == '__main__':
    train()