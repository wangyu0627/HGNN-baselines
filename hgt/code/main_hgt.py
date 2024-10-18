import numpy
import torch
from utils.load_data import load_data
# from model.HGT import HGT
from model.DGL_HGT import HGT
import warnings
import torch.nn as nn
import datetime
from utils.parser import parse_args
from utils.utils import EarlyStopping
from sklearn.metrics import f1_score, roc_auc_score
import os
import pickle as pkl
import random

def score(logits, labels):
    indices = torch.argmax(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func, predict_ntype): #, predict_ntype
    model.eval()
    with torch.no_grad():
        logits = model(g, features, predict_ntype)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
    auc_logits = nn.functional.softmax(logits)
    auc = roc_auc_score(y_true=labels[mask].detach().cpu().numpy(),
                        y_score=auc_logits[mask].detach().cpu().numpy(),multi_class='ovr')

    return loss, accuracy, auc, micro_f1, macro_f1

def main(i):
    warnings.filterwarnings('ignore')
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")

    ## random seed ##
    seed = args.seed
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    feats, label, idx_train, idx_val, idx_test = load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = int(max(label) + 1)
    # G processed
    G_file = open(os.path.join(args.data_path + args.dataset, args.dataset + "_hg.pkl"), "rb")
    G = pkl.load(G_file)
    G_file.close()
    predict_ntype = args.predict_ntype
    # nb_classes = label.shape[-1]
    features = {ntype: feats[i].cuda() for i, ntype in enumerate(G.ntypes)}
    node_dict = {}
    edge_dict = {}
    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        G.edges[etype].data["id"] = (
                torch.ones(G.num_edges(etype), dtype=torch.long) * edge_dict[etype]
        )
    G = G.to(device)
    # print("seed ", args.seed)
    # print("Dataset: ", args.dataset)
    # print("The number of meta-paths: ", len(G.canonical_etypes)/2)

    # model = HGT(args.ntypes, args.hidden_dim, nb_classes, args.num_heads, G.ntypes,
    #             G.canonical_etypes, predict_ntype, args.num_layers, args.dropout).to(device)
    model = HGT(args.ntypes, node_dict, edge_dict, args.ntypes, args.hidden_dim,
                nb_classes, args.num_layers, args.num_heads, args.dropout).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        # print('Using CUDA')
        model.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    stopper = EarlyStopping(patience=args.patience)

    for epoch in range(args.nb_epochs):
        logits = model(G, features, predict_ntype)
        loss = loss_fcn(logits[idx_train[i]], label[idx_train[i]])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[idx_train[i]], label[idx_train[i]])

        val_loss, val_acc, val_auc, val_micro_f1, val_macro_f1 = evaluate(
            model, G, features, label, idx_val[i], loss_fcn, predict_ntype) # , predict_ntype

        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        # print("Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
        #         "Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
        #     epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(),
        #     val_micro_f1, val_macro_f1))
        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_auc, test_micro_f1, test_macro_f1 = evaluate(
        model, G, features, label, idx_test[i], loss_fcn, predict_ntype) # , predict_ntype
    print(
        "\t[train_ratio] {:.4f} | Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test auc {:.4f}".format(
            i, test_loss.item(), test_micro_f1, test_macro_f1, test_auc))

    # 创建字符串
    formatted_string = "ratio:{}//seed:{}//lr:{}//weight_decay:{}//num_heads:{}//dropout:{}" \
                       "\nTest Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test auc {:.4f}".format(
        i, args.seed, args.lr, args.weight_decay, args.num_heads, args.dropout, test_micro_f1, test_macro_f1, test_auc)
    # 打开文件并写入字符串
    with open('results2.txt', 'a') as file:
        file.write(formatted_string + '\n')


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # for i in range(3):
    main(i=2)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")