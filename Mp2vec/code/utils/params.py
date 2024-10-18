import argparse
import sys


argv = sys.argv
dataset = 'dblp'


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument("--data_path", nargs="?", default="../data/", help="Input data path.")
    parser.add_argument('--target', type=str, default='paper')
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0006)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument("--data_path", nargs="?", default="../data/", help="Input data path.")
    parser.add_argument('--target', type=str, default='author')
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args


def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument("--data_path", nargs="?", default="../data/", help="Input data path.")
    parser.add_argument('--target', type=str, default='paper')
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--l2_coef', type=float, default=0)

    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def freebase_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)    
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument("--data_path", nargs="?", default="../data/", help="Input data path.")
    parser.add_argument('--target', type=str, default='movie')
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args


def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "aminer":
        args = aminer_params()
    elif dataset == "freebase":
        args = freebase_params()
    return args
