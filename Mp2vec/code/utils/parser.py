import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HAN")

    parser.add_argument("--seed", type=int, default=0, help="random seed for init")
    parser.add_argument(
        "--dataset",
        default="dblp",
        help="Dataset to use, default: acm",
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.0012)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_heads', type=list, default=[2])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    # parser.add_argument('--type_num', type=list, default=[4019, 7167, 60])
    # parser.add_argument('--type_num', type=list, default=[3492, 2502, 33401, 4459])
    # parser.add_argument('--type_num', type=list, default=[4057, 14328, 7723, 20])
    parser.add_argument('--type_num', type=list, default=[6564, 13329, 35890])
    return parser.parse_args()