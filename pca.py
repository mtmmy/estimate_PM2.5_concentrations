import csv
import numpy as np
from sklearn.decomposition import PCA

def loadCSV():
    records = []
    results = []
    with open("allPMData.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            records.append(list(map(float, row[:-1])))
            results.append(list(map(float, row[-1:])))
    return (np.array(records), np.array(results))

records, results = loadCSV()
pca = PCA(n_components=3)
newRecords = pca.fit_transform(records)
newData = np.append(newRecords, results, axis=1)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# print(sum(pca.explained_variance_ratio_))