import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class ChiSquareSelector(BaseEstimator,TransformerMixin):
    def __init__(self,alpha=0.05):
        self.alpha=alpha

    def fit(self, X,y):
        self.cols_to_keep_=[]
        for col in X.select_dtypes(include='object').columns:
            _, pval,_,__=chi2_contingency(pd.crosstab(X[col],y))
            if pval<=self.alpha:
                self.cols_to_keep_.append(col)
        self.other_cols_=X.select_dtypes(exclude='object').columns.tolist()
        return self
    def transform(self,X):
        return X[self.cols_to_keep_+self.other_cols_]



class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=6):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.cols_to_keep_ = []
        numeric_cols = X.select_dtypes(include='number').columns.tolist()
        total_cols = len(numeric_cols)
        vif_data = X[numeric_cols]
        cols_index = 0
        for col in range(0, total_cols):
            vif_value = variance_inflation_factor(vif_data, cols_index)
            if vif_value <= self.threshold:
                self.cols_to_keep_.append(numeric_cols[col])
                cols_index = cols_index + 1
            else:
                vif_data = vif_data.drop([numeric_cols[col]], axis=1)
        self.other_cols_ = X.select_dtypes(include='object').columns.tolist()
        return self

    def transform(self, X):
        return X[self.cols_to_keep_ + self.other_cols_]




class AnovaSelector(BaseEstimator,TransformerMixin):
    def __init__(self,alpha=0.05):
        self.alpha=alpha

    def fit(self, X,y):
        self.cols_to_keep_=[]
        numeric_cols=X.select_dtypes(include='number').columns.tolist()
        for col in numeric_cols:
            groups=[X[col][y==cat].dropna() for cat in  pd.unique(y)]
            f_stat,p_value=f_oneway(*groups)
            if p_value<=self.alpha:
                self.cols_to_keep_.append(col)
        self.other_cols_=X.select_dtypes(include='object').columns.tolist()
        return self
    def transform(self,X):
        return X[self.cols_to_keep_+self.other_cols_]


class EduEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,col):
        self.col=col

    def fit(self, X,y=None):
        self.education_level_map = {
            'SSC': 1,
            '12TH': 2,
            'UNDER GRADUATE': 3,
            'GRADUATE': 3,
            'PROFESSIONAL': 3,
            'POST-GRADUATE': 4,
            'OTHERS': 1
        }
        return self
    def transform(self,X):
        X=X.copy()
        X[self.col]=X[self.col].map(self.education_level_map)
        X[self.col]=X[self.col].astype('int64')
        return X




def build_onehot_step(cat_cols):
    return ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )
