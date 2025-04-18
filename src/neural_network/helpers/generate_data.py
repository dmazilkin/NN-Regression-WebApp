import numpy as np
import pandas as pd

def calculate_target(x):
    b = np.array([[0.2], [-1.31], [-0.9], [0.3]]).reshape(4,-1)
    r1 = np.sum((x-b)**2 - 10 * np.cos(0.7*np.pi*(x+b)), axis=0)
    r2 = np.sum((x+b*2.1)**2 - 5 * np.cos(0.2*np.pi*(x+b*4)), axis=0)
    return 40 + r1 + r2

def create_dataset(dataset_size: int) -> pd.DataFrame:
  high_boundary = 7.4
  min_boundary = -7.4
  feature_size = 4
  df = pd.DataFrame(columns=['f1', 'f2', 'f3', 'f4', 'target'])
  for _ in range(dataset_size):
    features = np.array([np.random.uniform(min_boundary, high_boundary) for _ in range(feature_size)])
    target = calculate_target(features.reshape(-1, 1))
    obj = [*features, *target]
    df.loc[len(df)] = obj
  return df

