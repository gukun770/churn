from helper import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score,  f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import PredefinedSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

class PreselectColumns(BaseEstimator, TransformerMixin):
    preselect_cols = list(set(['avg_dist',
 'avg_rating_by_driver',
 'avg_rating_of_driver',
 'avg_surge',
 'city',
 'last_trip_date',
 'phone',
 'signup_date',
 'surge_pct',
 'trips_in_first_30_days',
 'luxury_car_user',
 'weekday_pct',
 'churn']))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.loc[:, self.preselect_cols]

# class FilterRows(BaseEstimator, TransformerMixin):
#     """Only keep columns that have pct nan < 70%.
#     """
#     def fit(self, X, y):
#         percent = (X.isnull().sum()/X.isnull().count()*100).sort_values(ascending = False)
#         self.keep_columns = percent[percent<70].index.values
#         return self

#     def transform(self, X):
#         return X.loc[:, self.keep_columns]

class ReplaceNaN(BaseEstimator, TransformerMixin):

    num_col_name = ['avg_rating_by_driver','avg_rating_of_driver']
    cat_col_name = []

    def fit(self, X, y):

        self.dict = {}
        if self.cat_col_name:
            cat_mod = X[self.cat_col_name].mode().values.flatten()
            for col_name, value in zip(self.cat_col_name, cat_mod):
                self.dict[col_name] = value
        if self.num_col_name:
            num_median = X[self.num_col_name].median().values.flatten()
            for col_name, value in zip(self.num_col_name, num_median):
                self.dict[col_name] = value
        return self

    def transform(self, X):
        # print(report_nan(X))
        X.fillna(value=self.dict, inplace=True)
        return X

class BinRides(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self,X):
        if ('trips_in_first_30_days' in X.columns):
#newcols = pd.DataFrame(dict('zero':))
            X['zero_rides'] = (X['trips_in_first_30_days'] == 0)
            X['one_rides'] = X['trips_in_first_30_days'] == 1
            X['few_rides'] = ((6> X['trips_in_first_30_days']) == (X['trips_in_first_30_days']> 1)) 
            X['many_rides'] = (X['trips_in_first_30_days'] >= 6)
        return X


class Getdummies(BaseEstimator, TransformerMixin):
    selected_for_dummies = [
    'city',
    'phone',
   ]

    def __init__(self, pct=0.000, boo = True):
        self.pct = pct
        self.boo = boo

    def get_params(self, **kwargs):
        return {'boo': self.boo, 'pct':self.pct}

    def fit(self, X, y):
        cols_to_dummify = [col for col in self.selected_for_dummies if col in X.columns]
        d = {}
        for col in cols_to_dummify:
            d[col] = X[col].value_counts(1, dropna=not(self.boo)).to_dict()
        self.dict = d
        return self

    def transform(self, X):
        cols_to_dummify = [col for col in self.selected_for_dummies if col in X.columns]
        for col in cols_to_dummify:
            subcat_used = [subcat for subcat, percent in self.dict[col].items() if percent > self.pct]

            if len(subcat_used) == len(self.dict[col]):
                dummies = pd.get_dummies(X[col],prefix=col,dummy_na=self.boo)
                col_to_drop = str(col) + '_' + str(min(self.dict[col], key=self.dict[col].get))
                dummies.drop(col_to_drop, axis=1, inplace=True)
                X[dummies.columns] = dummies
            else:
                dummies = pd.get_dummies(X[col],prefix=col,dummy_na=self.boo)
                prefixed_subcat_used = [str(col) + '_' + str(item) for item in subcat_used]
                used_dummies = dummies[prefixed_subcat_used]

                X[used_dummies.columns] = used_dummies

        X.drop(cols_to_dummify, axis=1, inplace=True)
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    cols_to_drop = [
    'last_trip_date',
    'saledate_converted',
    'churn',
    'signup_date',
    'avg_rating_by_driver',
    'surge_pct',
    'True',
    ]

    def fit(self, X, y):
        return self

    def transform(self, X):
        for col in self.cols_to_drop:
            if col in X.columns:
                X.drop(col, axis=1, inplace=True)
        return X

class BinAvgRatDistance(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
    #for the avg rating of driver columns
        X['avg_rating_of_driver'].fillna(value = 0, inplace=True)
        df_avg_rating_of_driver = pd.cut(X['avg_rating_of_driver'], [0,0.5,4,5.1], include_lowest=True, labels=['avg_rating_of_driver_no_rating', 'avg_rating_of_driver_1-4', 'avg_rating_of_driver_4-5'])
        df_avg_rating_of_driver = pd.get_dummies(df_avg_rating_of_driver)
        df_avg_rating_of_driver.drop('avg_rating_of_driver_1-4', axis=1, inplace=True)
        X = X.join(df_avg_rating_of_driver)
        #for the average distance column
        df_avg_dist = pd.cut(X['avg_dist'], [0,3,10,161], include_lowest=True, labels=['avg_dist_0-3', 'avg_dist_3-10', 'avg_dist_10+'])
        df_avg_dist = pd.get_dummies(df_avg_dist)
        df_avg_dist.drop('avg_dist_10+', axis=1, inplace=True)
        X = X.join(df_avg_dist)
        #for the phone columns
        df_phone = pd.get_dummies(X['phone'], drop_first=True)
        X = X.join(df_phone)
        #for the city columns
        df_cities = pd.get_dummies(X['city'])
        df_cities.drop("King's Landing", axis=1, inplace=True)
        X = X.join(df_cities)
        #for the luxury car columns
        df_luxury_car = pd.get_dummies(X['luxury_car_user'], drop_first=True)
        X = X.join(df_luxury_car)
        return X


# class BinAvgRatDistance2(BaseEstimator, TransformerMixin):
#     def fit(self, X, y):
#         return self

#     def transform(self, X):
#         #for the avg rating of driver columns
#         X['avg_rating_of_driver'].fillna(value = 0, inplace=True)
#         df_avg_rating_of_driver = pd.cut(X['avg_rating_of_driver'], [0,0.5,4,5.1], include_lowest=True, labels=['avg_rating_of_driver_no_rating', 'avg_rating_of_driver_1-4', 'avg_rating_of_driver_4-5'])
#         df_avg_rating_of_driver = pd.get_dummies(df_avg_rating_of_driver)
#         df_avg_rating_of_driver.drop('avg_rating_of_driver_1-4', axis=1, inplace=True)
#         X = X.join(df_avg_rating_of_driver)
#         #for the phone columns
#         df_phone = pd.get_dummies(X['phone'], drop_first=True)
#         X = X.join(df_phone)
#         #for the city columns
#         df_cities = pd.get_dummies(X['city'])
#         df_cities.drop("King's Landing", axis=1, inplace=True)
#         X = X.join(df_cities)
#         #for the luxury car columns
#         df_luxury_car = pd.get_dummies(X['luxury_car_user'], drop_first=True)
#         X = X.join(df_luxury_car)
#         #for the surge pct columns
#         df_surgX.drop(['avg_rating_of_driver','avg_dist', 'phone', 'city', 'luxury_car_user', 'surge_pct'], axis=1, inplace=True)e_pct = pd.cut(X['surge_pct'], [0,0.4,30,101], include_lowest=True, labels=['surge_pct_0', 'surge_pct_1-30', 'surge_pct_30+'])
#         df_surge_pct = pd.get_dummies(df_surge_pct)
#         df_surge_pct.drop('surge_pct_30+', axis=1, inplace=True)
#         X = X.join(df_surge_pct)
#         X.drop(['avg_rating_of_driver', 'phone', 'city', 'luxury_car_user', 'surge_pct'], axis=1, inplace=True)
#         return X
