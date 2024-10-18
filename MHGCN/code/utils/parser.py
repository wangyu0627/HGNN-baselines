import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HAN")

    parser.add_argument("--seed", type=int, default=0, help="random seed for init")
    parser.add_argument(
        "--dataset",
        default="aminer",
        help="Dataset to use, default: acm",
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    # parser.add_argument('--type_num', type=list, default=[4019, 7167, 60]) # acm
    # parser.add_argument('--type_num', type=list, default=[3492, 2502, 33401, 4459]) # freebase
    # parser.add_argument('--type_num', type=list, default=[4057, 14328, 7723, 20]) # dblp
    parser.add_argument('--type_num', type=list, default=[6564, 13329, 35890]) # aminer
    return parser.parse_args()