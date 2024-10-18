import numpy
import torch
from utils import load_data, set_params, evaluate
from dgl.nn.pytorch import MetaPath2Vec
from torch.utils.data import DataLoader
from torch.optim import SparseAdam
import warnings
import datetime
import pickle as pkl
import os
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
    label, idx_train, idx_val, idx_test = load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    # feats_dim_list = [i.shape[1] for i in feats]
    G_file = open(os.path.join(args.data_path + args.dataset, args.dataset + "_hg.pkl"), "rb")
    G = pkl.load(G_file)
    G_file.close()
    print("seed ",args.seed)
    print("Dataset: ", args.dataset)
    # print("The number of meta-paths: ", P)
    
    model = MetaPath2Vec(G, ['ap', 'pc', 'cp', 'pa'], window_size=2, emb_dim=64, negative_size=3).to(device)
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
    dataloader = DataLoader(torch.arange(G.num_nodes(args.target)), batch_size=128,
                            shuffle=True, collate_fn=model.sample)
    for epoch in range(args.nb_epochs):
        for (pos_u, pos_v, neg_v) in dataloader:
            pos_u = pos_u.to(device)
            pos_v = pos_v.to(device)
            neg_v = neg_v.to(device)
            model.train()
            loss = model(pos_u, pos_v, neg_v)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            average_loss += loss.item()
        average_loss = average_loss / len(dataloader)
        print("loss ", average_loss)
        if average_loss < best:
            best = average_loss
            best_t = epoch
            cnt_wait = 0
            nids = torch.LongTensor(model.local_to_global_nid[args.target]).to(device)
            embeds = model.node_embed(nids)
            torch.save(embeds, 'Mp2vec_' + own_str + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    print('Loading {}th epoch'.format(best_t))
    embeds = torch.load('Mp2vec_' + own_str + '.pkl')
    embeds.requires_grad = False
    # embeds = model.get_embeds(feats, mps)

    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label,
                 nb_classes, device, args.dataset, args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    
    # if args.save_emb:
    f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
    pkl.dump(embeds.cpu().data.numpy(), f)
    f.close()


if __name__ == '__main__':
    train()
