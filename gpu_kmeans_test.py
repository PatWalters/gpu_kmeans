#!/usr/bin/env python

import sys
import pandas as pd
import cudf
from cuml import KMeans
import time

df = pd.read_csv("fp.csv")
cdf = cudf.DataFrame().from_pandas(df)
num_clusters = 1000
for num_clusters in range(1000,10000,1000):
    start = time.time()
    km = KMeans(n_clusters=num_clusters, n_gpu=-1)
    km.fit(cdf)
    elapsed = time.time() - start
    print(f"{num_clusters:10d} {elapsed:.2f}")
    
