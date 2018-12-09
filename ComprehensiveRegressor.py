import pandas as pd
import numpy as np
import math
import warnings 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost 
import lightgbm
from catboost import CatBoostRegressor
from sklearn import linear_model,svm, metrics,tree
from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import optunity
import optunity.metrics
#np.set_printoptions(threshold=np.inf)

warnings.filterwarnings('ignore')

class ComprehensiveRegressor:
    def __init__(self, data, nfolds,models = ['cb','xgb','gbm']):
        self.nfolds = nfolds
        self.models = models
        Y = data.AVG_PM_2_5
        # X = data.drop(['AVG_PM_2_5','Year','Month','Day','RAIN'], axis=1)
        X = data.drop(['AVG_PM_2_5'], axis=1)

        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(X, Y, train_size=0.9, random_state=1234)

        '''
        sc = StandardScaler()
        sc.fit(train_X)
        self.train_X = sc.transform(train_X)
        self.test_X = sc.transform(test_X)
        '''
    def stacking_train_test_split (self,data, pm_data, iteration_time):

        np_x_data = np.array(data)
        np_y_data = np.array(pm_data)

        start_flag = int((iteration_time)*0.2*(len(np_x_data)-1))
        end_flag = int((iteration_time+1.0)*0.2*(len(np_x_data)-1))


        if (iteration_time ==0):
            X_train = np_x_data[end_flag+ 1: ,:]  
            y_train = np_y_data[end_flag+ 1: ]
            X_test = np_x_data[0: end_flag,:]


        elif (iteration_time == 4):
            X_train = np_x_data[0: start_flag-1,:]  
            y_train = np_y_data[0: start_flag-1]
            X_test = np_x_data[start_flag: ,:]
        else :
            X_train = np.concatenate((np_x_data[0: start_flag-1,:], np_x_data[end_flag+ 1: ,:]), axis=0) 
        
            y_train = np.concatenate((np_y_data[0: start_flag-1], np_y_data[end_flag+ 1:]), axis=0) 

            X_test = np_x_data[start_flag: end_flag, :]    

        return X_train, y_train, X_test

    def xgboost_for_stack (self,X_train, y_train, X_test):
    
        #250
        xgb = xgboost.XGBRegressor(n_estimators=250, learning_rate=0.18, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=8,verbose_eval=True,silent=False)

        xgb.fit(X_train, y_train)
        predictions_xgb_train = xgb.predict(np.array(X_test))
        predictions_xgb_test = xgb.predict(np.array(self.test_X))

        return predictions_xgb_train.T , predictions_xgb_test

    def lightgbm_for_stack (self, X_train, y_train, X_test):
        ###########################light gbm###########################
        #500
        lgb = lightgbm.LGBMRegressor(n_estimators=500, boosting_type= 'gbdt', objective= 'regression', metrics= 'rmsle', max_depth= 15, learning_rate= 0.2, verbose=0,num_leaves=127)
    
        lgb.fit(X_train, y_train)
    
        predictions_lgb_train = lgb.predict(np.array(X_test))   
        predictions_lgb_test = lgb.predict(np.array(self.test_X))  
        ##################################################################
    
        return predictions_lgb_train, predictions_lgb_test

    def stacking_train(self,X_train_pre, y_train_pre, X_test_pre, model):
        for i in range(5):
            X_train_stack, Y_train_stack, X_test_stack = self.stacking_train_test_split(X_train_pre, y_train_pre, i)
            if i == 0 :
                if model == 'xgb':
                    stacking_train, stacking_test = self.xgboost_for_stack(X_train_stack, Y_train_stack, X_test_stack)
                else:
                    stacking_train, stacking_test = self.lightgbm_for_stack(X_train_stack, Y_train_stack, X_test_stack)
            else :
                if model == 'xgb':
                    fold_train, fold_test = self.xgboost_for_stack(X_train_stack, Y_train_stack, X_test_stack)
                else:
                    fold_train, fold_test = self.lightgbm_for_stack(X_train_stack, Y_train_stack, X_test_stack)
                stacking_train = np.hstack((stacking_train, fold_train.T))
                stacking_test = stacking_test + fold_test

        return stacking_train, stacking_test/5.0

    def train(self):
        self.models_kf_return = {}

        # ------------------------------------------------------------------- Layer 1 -------------------------------------------------------------------
        kf = KFold(n_splits = self.nfolds, shuffle=False, random_state=0)

        kf_train = np.zeros(self.train_X.shape[0]) #生成训练集预测结果的容器
        kf_test = np.zeros(self.test_X.shape[0]) #生成测试集预测结果的容器
        kf_test_predict = np.empty([self.nfolds,self.test_X.shape[0]]) #生成测试集多折预测结果的容器
        
        # removed less important feature
        # cbData = self.train_X.drop(['IPREC','Hour'], axis=1)
        cbData = self.train_X
        xgData = np.copy(self.train_X)
        gbmData = np.copy(self.train_X)
        # ----------------------------------  CatBoost Region ----------------------------------
        for i, (train_index, test_index) in enumerate(kf.split(cbData)):
            x_tr = cbData.iloc[train_index]
            x_te = cbData.iloc[test_index]
            y_tr = self.train_Y.iloc[train_index]
            #500
            clf = CatBoostRegressor(iterations=500, depth=16, learning_rate=0.1, loss_function='RMSE')

            clf.fit(x_tr, y_tr) #折数训练

            kf_train[test_index] = clf.predict(x_te) #第i折中训练预测结果保存
            
            kf_test_predict[i, :] = clf.predict(self.test_X) #预测测试集

        kf_test[:] = kf_test_predict.mean(axis=0) #取平均
        
        cb_kf_train = kf_train.reshape(-1, 1)
        cb_kf_test = kf_test.reshape(-1, 1)

        self.models_kf_return['cb'] = [cb_kf_train, cb_kf_test]
        # ---------------------------------- CatBoost Region ----------------------------------

        stacked_x_train, stacked_y_test = self.stacking_train(xgData, self.train_Y, self.test_X,'xgb')
        self.models_kf_return['xgb'] = [stacked_x_train.reshape(-1,1), stacked_y_test.reshape(-1,1)]

        stacked_x_train_lgb, stacked_y_test_lgb = self.stacking_train(gbmData, self.train_Y, self.test_X, 'gbm')
        self.models_kf_return['gbm'] = [stacked_x_train_lgb.reshape(-1,1), stacked_y_test_lgb.reshape(-1,1)]

        # ---------------------------------- Data Concatenate ---------------------------------- 
        for i in range(len(self.models)):
            if i == 0:
                self.final_train = self.models_kf_return[self.models[i]][0]
                self.final_test = self.models_kf_return[self.models[i]][1]
            else:
                self.final_train = np.concatenate((self.final_train,self.models_kf_return[self.models[i]][0]), axis=1)
                self.final_test = np.concatenate((self.final_test,self.models_kf_return[self.models[i]][1]), axis=1)

        # ---------------------------------- Data Concatenate ---------------------------------- 
    def predict(self):
        # Linear Regression
        
        regr = linear_model.LinearRegression()

        tt =  np.concatenate((self.final_train.reshape(-1,3),np.array(self.train_Y).reshape(-1,1)), axis=1) 
        np.savetxt("input.csv",tt,delimiter=",")
        np.savetxt("testData.csv",np.array(self.final_test).reshape(-1,3),delimiter=",")
        np.savetxt("testLabel.csv",np.array(self.test_Y).reshape(-1,1),delimiter=",")
        
        # Train the model using the training sets
        regr.fit(self.final_train.reshape(-1,3), np.array(self.train_Y).reshape(-1,1))

        # Make predictions using the testing set
        predictions_ligr = regr.predict(np.array(self.final_test).reshape(-1,3))

        print("RMSE: %.2f" % math.sqrt(np.mean((predictions_ligr - np.array(self.test_Y).reshape(-1,1)) ** 2)))
        
        # below is the svm provided by Allan
        '''
        ss = svm.SVR(C=1.0, epsilon=0.25)
        ss.fit(self.final_train.reshape(-1,2),np.array(self.train_Y).reshape(-1,1))

        predictions = ss.predict(np.array(self.final_test).reshape(-1,2))

        predictions=predictions.reshape(-1, 1)

        scores=optunity.metrics.mse(np.array(self.test_Y).reshape(-1,1), predictions)

        RM=math.sqrt(scores)
        print(RM)
        '''


