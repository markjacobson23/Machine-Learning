import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import set_config

set_config(transform_output="pandas")

def add_family_features(X):
    X= X.copy()
    X['Fsize'] = X['SibSp'] + X['Parch'] + 1
    X['Fcost'] = X['Fsize'] * X['Fare']

    return X
family_features = FunctionTransformer(add_family_features, validate=False)


numeric_column_addition = Pipeline([
    ("add_family_features",family_features),
])

numeric_transformer = Pipeline([
    ("num_imputer", SimpleImputer(strategy="median")),
    ("add_numeric_columns",numeric_column_addition),
    ("scaler", StandardScaler())
])


class CategoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,param=None):
        self.param=param

    def fit(self, X, y=None):
        title_freq = X['Name'].str.extract(r",\s*([^\.]+)\.").value_counts()
        self.common_titles = set((title_freq >= 5).index)
        return self

    def transform(self, X):
        X = X.copy()
        titles = X['Name'].str.extract(r",\s*([^\.]+)\.")
        X['RareTitle'] = ~titles.isin(self.common_titles).astype(int)
        X['Title'] = titles
        X['Deck'] = X['Cabin'].str[0]
        X=X.drop(['Name','Cabin'],axis=1)
        return X


categorical_transformer = Pipeline([
    ("cat_imputer", SimpleImputer(strategy='most_frequent')),
    ('cat_adder',CategoryTransformer()),
    ('encoder',OneHotEncoder(handle_unknown='infrequent_if_exist',drop='if_binary',sparse_output=False,min_frequency=5))
])

numeric_features = ['Pclass','Age','SibSp','Parch','Fare']
categorical_features = ['Name','Sex','Cabin','Embarked']

preprocessor = ColumnTransformer(
    transformers = [
        ('numeric',numeric_transformer, numeric_features),
        ('categorical',categorical_transformer, categorical_features)
    ],
)

Model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(C=10, solver='lbfgs', max_iter=20000))
])
train_data = pd.read_csv('../data/train.csv')
train_targets = train_data['Survived']
test_data = pd.read_csv('../data/test.csv')
test_indexes = test_data['PassengerId'].to_numpy().flatten()
Model.fit(train_data,train_targets)
submission = pd.DataFrame({
    "PassengerId": test_indexes,
    "Survived": Model.predict(test_data)
})





