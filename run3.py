import numpy as np
import pandas as pd
import umap
from DFKM import DeepFuzzyKMeans

csv_data = pd.read_csv("3000matrix.csv")  # 读取训练数据
# dat = umap.UMAP(n_neighbors=20, min_dist=0.3, n_components=100).fit_transform(csv_data)
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
dat = pca.fit_transform(csv_data)
dat = np.array(dat)

print(dat.shape)
csv_label = pd.read_csv("cluster_labels_151673.csv")  # 读取
labe = np.array(csv_label)
labee = [int(x) for item in labe for x in item]


print(labe.shape)
if __name__ == '__main__':
    #import data_loader as loader
    data, labels = dat, labee
    data = data.T
    for lam in [0.01]: #10**-3
        print('lam={}'.format(lam))
        dfkm = DeepFuzzyKMeans(data, labels, [data.shape[0], 512, 300], lam=lam, gamma=1, batch_size=512, lr=10**-4)
        dfkm.run()
        lab = dfkm.labels
        print(lab)
        dataCSV = pd.DataFrame(lab)
        dataCSV.to_csv("parelab.csv", index=False, mode="a", header=False, encoding="GBK")


# dataCSV=np.array(dataCSV)
# print(type(dataCSV))
#
# dataCSV1 = [int(x) for item in dataCSV for x in item]
# print(type(dataCSV1))