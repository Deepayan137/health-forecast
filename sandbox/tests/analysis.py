import os
import pdb
import numpy as np
import pandas as pd

from sandbox.data_utils import get_groundtruth

if __name__ == "__main__":
	for trial in range(1, 21):
		pred_file = os.path.join("sandbox", "saved", f"pred_{trial}.npy")
		y_pred = np.load(pred_file)
		gt_file = os.path.join("sandbox", "data", "phds", f'truesetpoint_{trial}.csv')
		y_true = get_groundtruth(gt_file)
		pdb.set_trace()