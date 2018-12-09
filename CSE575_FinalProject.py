import pandas as pd
import warnings
import ComprehensiveRegressor as CR
from sklearn.decomposition import PCA
import numpy as np
import csv
warnings.filterwarnings('ignore')


train = pd.read_csv("allPMData.csv")
features = ["Hour","Season","DEWP","HWMI","PRES","TEMP","CBWD","IWS","IPREC","City"]
x = train.loc[:, features].values
y = train.loc[:, ["AVG_PM_2_5"]]

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, train[['AVG_PM_2_5']]], axis = 1)

model =CR.ComprehensiveRegressor(finalDf, 5, ['cb','xgb','gbm'])

model.train()
model.predict()
