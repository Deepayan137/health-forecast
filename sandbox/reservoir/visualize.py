import argparse

from reservoirpy.hyper import plot_hyperopt_report


parser = argparse.ArgumentParser()
parser.add_argument('--study_name', type=str, required=True)
args = parser.parse_args()

fig = plot_hyperopt_report(args.study_name, ("lr", "sr"), metric="loss")
fig.savefig('sample.jpg')