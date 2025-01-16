import pandas as pd
from tqdm import tqdm

from src.graph import KNNGraph

df = pd.read_csv("data/csg/raw/CSG.csv")
knn = KNNGraph(rcut=7.5)

n = 50_000
for i, row in tqdm(df.iterrows(), total=n):
    data = knn.convert(row["Structure"])

    if i == n:
        break
