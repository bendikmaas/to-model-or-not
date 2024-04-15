import argparse

import panda as pd
import seaborn as sns

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment-id", "-e", required=True)
  args = parser.parse_args()
  
  experiment = tb.data.experimental.ExperimentFromDev(args.experiment_id)
  df = experiment.get_scalars()