import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K=1
for seed in range(1,5):
    mixture, post = common.init(X=X, K=K, seed=seed)
    mixture, post, cost = kmeans.run(X=X, mixture=mixture, post=post)
    print(seed, cost)