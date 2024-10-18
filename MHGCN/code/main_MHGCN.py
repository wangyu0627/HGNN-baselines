import numpy
import torch
from utils.load_data import load_data
from model.model import MHGCN
import warnings
import torch.nn as nn
import datetime
from utils.parser import parse_args
from utils.utils import EarlyStopping
from sklearn.metrics import f1_score, roc_auc_score
import dgl
import random

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(features, g)
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

    feats, mps, label, idx_train, idx_val, idx_test = load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = int(max(label) + 1)
    # nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]

    P = int(len(mps))
    # print("seed ", args.seed)
    # print("Dataset: ", args.dataset)
    # print("The number of meta-paths: ", P)

    model = MHGCN(P, feats_dim_list[0], args.num_layer, nb_classes, args.dropout).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        # print('Using CUDA')
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        gs = [g.to(device) for g in mps]
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    stopper = EarlyStopping(patience=args.patience)

    for epoch in range(args.nb_epochs):
        logits = model(feats[0], gs)
        loss = loss_fcn(logits[idx_train[i]], label[idx_train[i]])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[idx_train[i]], label[idx_train[i]])

        val_loss, val_acc, val_auc,val_micro_f1, val_macro_f1 = evaluate(
            model, gs, feats[0], label, idx_val[i], loss_fcn)

        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        # print("Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
        #         "Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
        #     epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(),
        #     val_micro_f1, val_macro_f1))
        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_auc, test_micro_f1, test_macro_f1 = evaluate(
        model, gs, feats[0], label, idx_test[i], loss_fcn)
    print(
        "\t[train_ratio] {:.4f} | Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test auc {:.4f}".format(
            i, test_loss.item(), test_micro_f1, test_macro_f1, test_auc))

    # 创建字符串
    formatted_string = "ratio:{}//seed:{}//lr:{}//weight_decay:{}//dropout:{}" \
                       "\nTest Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test auc {:.4f}".format(
        i, args.seed, args.lr, args.weight_decay, args.dropout, test_micro_f1, test_macro_f1, test_auc)
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