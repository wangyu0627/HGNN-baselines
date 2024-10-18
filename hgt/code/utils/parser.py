import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HAN")
    dataset = "dblp"
    parser.add_argument("--seed", type=int, default=0, help="random seed for init")
    parser.add_argument(
        "--dataset",
        default=dataset,
        help="Dataset to use, default: acm",
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument("--data_path", nargs="?", default="../data/", help="Input data path.")
    if dataset == 'acm':
        parser.add_argument('--predict_ntype', type=str, default='paper') #acm
        parser.add_argument('--ntypes', type=dict, default={'author': 7167, 'paper': 1902, 'subject': 60})
        parser.add_argument('--type_num', type=list, default=[4019, 7167, 60]) #acm
    elif dataset == 'freebase':
        parser.add_argument('--predict_ntype', type=str, default='movie')  # freebase
        parser.add_argument('--ntypes', type=dict, default={'actor': 33401, 'direct': 2502, 'movie': 3492, 'writer': 4459})  # freebase
        parser.add_argument('--type_num', type=list, default=[3492, 2502, 33401, 4459]) #freebase
    elif dataset == 'dblp':
        parser.add_argument('--predict_ntype', type=str, default='author')  # dblp
        parser.add_argument('--ntypes', type=dict, default={'author': 334, 'conference': 20, 'paper': 4231, 'term': 7723})  # dblp
        parser.add_argument('--type_num', type=list, default=[4057, 14328, 7723, 20]) # dblp
    else:
        parser.add_argument('--predict_ntype', type=str, default='paper')  # aminer
        parser.add_argument('--ntypes', type=dict,default={'author': 13329, 'paper': 6564, 'reference': 35890}) # aminer
        parser.add_argument('--type_num', type=list, default=[6564, 13329, 35890]) # aminer
    return parser.parse_args()