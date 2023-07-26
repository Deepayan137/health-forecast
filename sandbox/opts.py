import argparse

def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--csv", type=str, default="sandbox/data/simulated_data_4reservoirpy.csv")
    parser.add_argument("--err", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=2)
    parser.add_argument('--reduce_n', action='store_true', help='Reduce N (remove patient records)')
    parser.add_argument('--replace_t', action='store_true', help='Replace T with NA (remove data on specific days and replace with NA)')
    parser.add_argument("--rr_n", type=float, default=0.9)
    parser.add_argument("--rr_t", type=float, default=0.1)
    parser.add_argument("--record_every", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="simulated")
    parser.add_argument("--imp_method", type=str, default="saits")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--forecast", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--out_dim", type=int, default=5)
    parser.add_argument("--lstm_epochs", type=int, default=10)
    parser.add_argument("--seed_timesteps", type=int, default=10)
    opt = parser.parse_args()
    return opt