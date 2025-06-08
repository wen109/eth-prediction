# ETH Price Prediction using XGBoost with Feature Selection and Hyperparameter Optimization
# This script fetches historical ETH price data, applies technical indicators,
# performs feature selection, hyperparameter tuning, and evaluates the model's performance.     
# It also includes methods for visualizing results such as confusion matrix and ROC curve.
# Author: [Ven]
# Date: [2023-10-01]
# License: [MIT License]
# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, 
                                     RandomizedSearchCV,
                                     TimeSeriesSplit
                                    )
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, to_graphviz
from sklearn.metrics import (accuracy_score,
                            auc,
                            roc_curve,
                            RocCurveDisplay,
                            ConfusionMatrixDisplay,
                            confusion_matrix
                            )
from sklearn.metrics import(classification_report,
                           confusion_matrix)
import ccxt
import os
from fear_and_greed import FearAndGreedIndex
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier



class eth_predict_optimized():
    def __init__(self,symbol,timeframe,selecttion_run_name='optimized_wrapper_embedded_mix'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.selecttion_run_name =selecttion_run_name
        self.exchange =ccxt.binance()
        self.fng =FearAndGreedIndex()
        self.sentiment_cache =None
        self.xgb_cache =None
        self.features_cache =None
        self.features_wrapper_cache =None
        self.features_embedded_cache =None
        self.hyperparams_optimized_cache =None
        self.params_grid ={
            'max_depth':[3,4],
            'learning_rate':[0.01,0.05,0.2],
            'n_estimators':[50,80,100],
            'subsample': [0.6,0.8,1],
            'colsample_bytree': [0.6,0.8,1],
            'gamma': [0.04,0.05],
            'min_child_weight': [0.8,1,1.2]
        }
        self.optimized_model_cache= None
        
    def obtain_df(self):
        
        exchange =self.exchange
        symbol = self.symbol
        timeframe = self.timeframe
        since =exchange.parse8601('2017-01-01T00:00:00Z')
        limit = 1000
        df =[]
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol,
                                         timeframe=timeframe,
                                        since = since,
                                        limit = 1000)
            if not ohlcv:
                break
            df +=ohlcv
            since = ohlcv[-1][0]+1
        
        df = pd.DataFrame(df,columns = ['timestamp','open','high',
                                             'low','close','volume'])
        df['timestamp']=pd.to_numeric(df['timestamp'],errors='coerce')
        df['date']=pd.to_datetime(df['timestamp'],unit='ms').dt.tz_localize(None)
        
        df =df[['date','open','high','low','close','volume']]
        
        
        return df
    
    
    def append_sentiment(self):
        if self.sentiment_cache is not None: 
            return self.sentiment_cache 
        fng =self.fng
        fng_data = fng.get_historical_data(datetime.now()-timedelta(days=365*8))
        fng_df = [(item['timestamp'],item['value']) for item in fng_data]
        fng_df =pd.DataFrame(fng_df, columns =['timestamp','value'])
        fng_df['timestamp']=pd.to_numeric(fng_df['timestamp'],errors='coerce')
        fng_df['date'] =pd.to_datetime(fng_df['timestamp'],unit='s').dt.tz_localize(None)
        fng_df['value']=pd.to_numeric(fng_df['value'],errors='coerce')
        fng_df=fng_df[['date','value']] 
        fng_df=fng_df.set_index('date')

        
        df = self.obtain_df()
        df = df.set_index('date')

        df=df.merge(fng_df,right_index=True,left_index=True, how='left')
        df=df.dropna()
        self.sentiment_cache =df
        return df
    
    def atr(self,period=14):
        df =self.append_sentiment()
        high=df['high']
        low=df['low']
        close=df['close']
        tr =pd.concat([high-low,(high-close.shift()).abs(),
                       (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return pd.DataFrame({'atr':atr})
    
    def bbands(self,window=20,stds=2):
        df =self.append_sentiment()
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        upper =sma +stds*std
        lower =sma-stds*std
        
        return pd.DataFrame({'bb_lower':lower,'bb_mid':sma,'bb_upper':upper})
    
    def macd(self,fast=12,slow=26, signal=9):
        df =self.append_sentiment()
        ema_fast = df['close'].ewm(span=fast,adjust=False).mean()
        ema_low = df['close'].ewm(span=slow,adjust=False).mean()
        macd_line =ema_fast-ema_low
        signal_line =macd_line.ewm(span=signal).mean()
        histogram =macd_line -signal_line
        
        return  pd.DataFrame({'macd_line':macd_line,
                              'signal_line':signal_line, 
                             'histogram':histogram})
        
    def rsi(self,period=14):
        df =self.append_sentiment()
        delta =df['close'].diff()
        
        gain =delta.where(delta>0,0.0)
        loss =-delta.where(delta<0,0.0)
        
        avg_gain =gain.rolling(window=period).mean()
        avg_loss =loss.rolling(window=period).mean()
        
        rs =avg_gain/avg_loss
        rsi =100-100/(rs+1)
        return pd.DataFrame({'rsi':rsi})
    
    def features_list(self):
        df= self.append_sentiment()
        for i in range(5,50,5):
            df['SMA_'+str(i)]=df['close'].rolling(window=i).mean()
            df['EMA'+str(i)]=df['close'].ewm(span=i,adjust=False).mean()
        df = pd.concat([df,self.atr()],axis=1)
        df = pd.concat([df,self.macd()[['macd_line']]],axis=1)
        df = pd.concat([df,self.bbands()],axis=1)
        df =pd.concat([df,self.rsi()],axis=1)
        
        df.dropna(inplace= True)
        
        x=df
        x= df.drop(['open', 'high', 'low','close'],axis=1)
        
        return x, df
        
        
    def target_var(self):
        df =self.features_list()[1]
        df['target']=np.where(df['close'].shift(-1)>1.005*df['close'],1,0)
        y =df['target']
        return y
    def features_filter(self, x_trained, x_test, y,variance_thresh=0.01,
                       mi_thresh=0.005,
                       corr_thresh=0.95):
        if self.features_cache is not None:
            return self.features_cache
        
        
        vt =VarianceThreshold(threshold =variance_thresh)
        x_var =pd.DataFrame(vt.fit_transform(x_trained),
                            columns =x_trained.columns[vt.get_support()])
        x_trained= x_var.copy()
        mi_scores =pd.Series(mutual_info_classif(x_trained,y), index= x_trained.columns)
        mi_selected =mi_scores[mi_scores >mi_thresh].index
        x_mi = x_trained[mi_selected]
        x_trained = x_mi.copy()
        
#         corr_matrix = x_trained.corr().abs()
#         upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 
#                                           k =1).astype(bool))
#         col_drop =[col for col in x_trained.columns if any(upper[col]>corr_thresh)]
#         x_corr =x_trained.drop(columns =col_drop)
#         x_trained = x_corr.copy()
        
        selected_features =x_trained.columns.to_list()
        x_test =x_test[selected_features]
        x_train_scaled, x_test_scaled =self.scalling(x_trained,x_test)
        
        self.features_cache = x_train_scaled, x_test_scaled, selected_features
        return x_train_scaled, x_test_scaled, selected_features
    
    def features_wrapper(self,x_train_scaled,x_test_scaled,y):
        if self.features_wrapper_cache is not None:
            return self.features_wrapper_cache
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        
        selector = SequentialFeatureSelector(
            model, direction='forward', scoring='roc_auc', n_jobs=-1, cv=5
        )
        selector.fit(x_train_scaled, y)
        x_trained, x_test = self.split_scalling()[0:2]
        x_train_scaled = x_train_scaled[:,selector.get_support()]
        x_test_scaled =x_test_scaled[:,selector.get_support()]
        features_selected =x_trained.loc[:,selector.get_support()].columns.to_list()

        self.features_wrapper_cache =x_train_scaled, x_test_scaled,features_selected
        return x_train_scaled, x_test_scaled,features_selected
    
    def features_embedded(self,x_train,x_test,y,top_n=10):
        path_name ='figures/embedded'
            
        if self.features_embedded_cache is not None:
            return self.features_embedded_cache
        
        
        model =XGBClassifier( use_label_encoder = False, eval_metric='logloss')
        model.fit(x_train,y)
        
        importance_df = pd.DataFrame({
            'feature': x_train.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending =False)
        
        selected_features = importance_df.head(top_n)['feature'].to_list()
        x_train =x_train[selected_features].copy()
        x_test =x_test[selected_features].copy()
        plt.barh(importance_df['feature'][:top_n][::-1],
                importance_df['importance'][:top_n][::-1])
        plt.title('XGBoost Embedded Feature Importance(Top 10)')
        plt.xlabel('feature importance')
        os.makedirs(path_name,exist_ok =True)
        plt.savefig(os.path.join(path_name,'pie_chart.png'))
        
        x_train_scaled, x_test_scaled= self.scalling(x_train, x_test)
       
        self.features_embedded_cache =x_train_scaled,x_test_scaled,selected_features
        
        return x_train_scaled,x_test_scaled,selected_features

    def scalling(self,x_train,x_test):
        
        scaler =StandardScaler()
        x_train_scaled =scaler.fit_transform(x_train)
        x_test_scaled =scaler.transform(x_test)
        
        return x_train_scaled, x_test_scaled
    def split_scalling(self):
    
        x= self.features_list()[0]
        y =self.target_var() 
        x_train, x_test, y_train, y_test  =train_test_split(x,y, test_size=0.2, shuffle=False)

        scaler=StandardScaler()
        x_train_scaled =scaler.fit_transform(x_train)
        x_test_scaled =scaler.transform(x_test)
        features_selected =x_train.columns
       
        return x_train, x_test, y_train, y_test,x_train_scaled, x_test_scaled, features_selected
    
    def cwts(self):
      
        x_test,y_train, y_test =self.split_scalling()[1:4]
        x_train_scaled =self.selection_run()[0]
        x_test_scaled =self.selection_run()[1]
        features_selected =self.selection_run()[2]

                                     
        y0, y1 =np.bincount(y_train)
        w0=(1/y0)*(len(y_train))/2
        w1=(1/y1)*(len(y_train))/2

        train_weights =[w0 if item==0 else w1 for item in y_train]
        dtrain =xgb.DMatrix(x_train_scaled, label =y_train, 
                            nthread=4, weight =train_weights,
                           feature_names =features_selected)

        test_weights=[w0 if item==0 else w1 for item in y_test]
        dtest =xgb.DMatrix(x_test_scaled, label =y_test, 
                           nthread=4, weight =test_weights,
                          feature_names =features_selected)
        return dtrain, dtest
    def xgb_classifier(self, Objective='binary:logistic',Eval_metric ='logloss'):
        Xgb_classifier = XGBClassifier(
            objective=Objective,
            eval_metric=Eval_metric,)
        
        return Xgb_classifier
    
    def tscv(self, N_splits=5, Gap=1):
        Tscv=TimeSeriesSplit(n_splits=N_splits, gap=Gap)
        
        return Tscv
    def hyperparams_tunning(self):
        if self.hyperparams_optimized_cache is not None:
            return self.hyperparams_optimized_cache
        xgb_classifer =self.xgb_classifier()
        params_grid=self.params_grid
        tscv=self.tscv()
        x_train_scaled =self.selection_run()[0]
        y_train =self.split_scalling()[2]
        
        random_search = RandomizedSearchCV(
            estimator =xgb_classifer,
            param_distributions =params_grid,
            n_iter =500,
            scoring='roc_auc',
            cv=tscv,
            verbose=10,
            n_jobs=-1)
        
        
        random_search.fit(x_train_scaled,y_train)
        
        hyperparams_optimized =random_search.best_params_
        
        self.hyperparams_optimized_cache=random_search.best_params_
        
        return hyperparams_optimized
    
    def optimized_model(self):
        if self.optimized_model_cache is not None:
            return self.optimized_model_cache
        optimized_params =self.hyperparams_tunning()
        optimized_params['eta']=  optimized_params.pop('learning_rate')
        optimized_params['objective']='binary:logistic'
        optimized_params['eval_metric']='logloss'
        dtrain,dtest =self.cwts()
        optimized_model =xgb.train(
            optimized_params,
            dtrain,
            num_boost_round =optimized_params.pop('n_estimators'),
            evals= [(dtrain,'train'),(dtest,'eval')],
            #early_stopping_rounds=20,
            verbose_eval =5
        )
        y_proba =optimized_model.predict(dtest)
        y_pred = np.round(y_proba)
        
        y_pred_train =np.round(optimized_model.predict(dtrain))
        self.optimized_model_cache =y_pred, y_pred_train, optimized_model
        return y_pred, y_pred_train, optimized_model
    
    def selection_run(self):
        if self.selecttion_run_name =='filter':
            x_train,x_test,y= self.split_scalling()[0:3] 
            x_train_scaled,x_test_scaled,features_selected =self.features_filter(x_train,x_test,y)
            
        elif self.selecttion_run_name =='wrapper':
            x_train_scaled,x_test_scaled=self.split_scalling()[4:6]
            y=self.split_scalling()[2]
            
            x_train_scaled,x_test_scaled,features_selected =self.features_wrapper(x_train_scaled,x_test_scaled,y)
            
        elif self.selecttion_run_name =='embedded':
            x_train,x_test,y= self.split_scalling()[0:3]
            x_train_scaled,x_test_scaled,features_selected =self.features_embedded(x_train,x_test,y)
        else:
            x_train_scaled,x_test_scaled=self.split_scalling()[4:6]
            y=self.split_scalling()[2] 
            features_selected =self.features_wrapper(x_train_scaled,x_test_scaled,y)[2]
            x_train,x_test,y= self.split_scalling()[0:3]
            x_train =x_train.loc[:,features_selected]
            x_test=x_test.loc[:,features_selected]
            x_train_scaled,x_test_scaled,features_selected =self.features_embedded(x_train,x_test,y,top_n=15)
            
        return x_train_scaled,x_test_scaled, features_selected
    
    def draw_confusion(self):
        selecttion_run_name =self.selecttion_run_name
        path_name ='figures/confusion'
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        
        y_pred =self.optimized_model()[0]
        y_test =self.split_scalling()[3]
        cm =confusion_matrix(y_test,y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down (0)', 'Up (1)'])
        disp.plot(cmap='Blues')
        plt.title(f'{selecttion_run_name}_Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(path_name,selecttion_run_name+'.png'),
                    dpi=100, bbox_inches ='tight')
        return plt.show()
    
    def draw_roc(self):
        selecttion_run_name =self.selecttion_run_name
        path_name ='figures/roc/'
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        y_pred= self.optimized_model()[0]
        y_test=self.split_scalling()[3]
        fpr, tpr, _ =roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        disp.plot()

        plt.title(f'{selecttion_run_name}_ROC Curve')
        plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(path_name,selecttion_run_name+'.png'),
                    dpi=300,bbox_inches='tight')
        return plt.show()
        
    def accuracy_check(self):
        y_train, y_test =self.split_scalling()[2:4]
        dtrain =self.cwts()[0]
        
        # Calculate training accuracy
        train_preds = self.optimized_model()[1]
        train_accuracy = accuracy_score(y_train, train_preds)

        # Calculate test accuracy
        test_preds = self.optimized_model()[0]
        test_accuracy = accuracy_score(y_test, test_preds)

        return [print(f"Training Accuracy: {train_accuracy:.4f}"),
        print(f"Test Accuracy: {test_accuracy:.4f}"),
        print(f"Difference (Training - Test): {train_accuracy - test_accuracy:.4f}")]
    

        
    
handle = eth_predict_optimized('ETH/USDT','1d','optimized_wrapper_embedded_mix')



