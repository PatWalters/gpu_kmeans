#!/usr/bin/env python

import numpy as np
import pandas as pd
import cudf
from cuml import KMeans
from cuml import PCA as cumlPCA


def np2cudf(arr):
    # convert numpy array to cudf dataframe
    df = pd.DataFrame({'fea%d'%i:arr[:,i] for i in range(arr.shape[1])})
    pdf = cudf.DataFrame()
    for c,column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf

def generate_fp_array(rows,cols):
    # generate random "cols" bit vectors with "rows" bits
    fp_array = np.random.randint(0,2,[rows,cols])
    fp_array = np.array(fp_array,dtype=np.float32)
    return fp_array

def do_pca(cudf_in,n_components):
    print("PCA")
    whiten = False
    random_state = 42
    svd_solver="full"
    pca_cuml = cumlPCA(n_components=n_components,svd_solver=svd_solver,
                       whiten=whiten, random_state=random_state)
    pc_df = pca_cuml.fit_transform(cudf_in)
    print("Done")
    print(pca_cuml.explained_variance_)
    return pc_df

def do_kmeans(cudf_in,num_clusters):
    print("KMeans")
    km = KMeans(n_clusters=num_clusters, n_gpu=-1)
    km.fit(cudf_in)
    print("Done")

def main():
    rows = 250000
    cols = 256
    fp_arr = generate_fp_array(rows,cols)
    print(fp_arr.shape)
    gpu_df = np2cudf(fp_arr)
    print(gpu_df.shape)
    pc_df = do_pca(gpu_df,50)
    print(pc_df.shape)
    do_kmeans(pc_df,10000)


if __name__ == "__main__":
    main()
