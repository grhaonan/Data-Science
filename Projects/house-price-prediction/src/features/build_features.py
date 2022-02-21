import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


PROCESSED_DATA_PATH = '../../data/processed/project_house_price_prediction'
PREPARED_DATA_PATH = PROCESSED_DATA_PATH

processed_training_set_path = os.path.join(PROCESSED_DATA_PATH, 'strat_train_set_features.csv')
processed_test_set_path = os.path.join(PROCESSED_DATA_PATH, 'strat_test_set_features.csv')

prepared_training_set_path = os.path.join(PROCESSED_DATA_PATH, 'strat_train_set_features_prepared.csv')
prepared_test_set_path = os.path.join(PROCESSED_DATA_PATH, 'strat_test_set_features_prepared.csv')

# load data from processed datasets
processed_training_set = pd.read_csv(processed_training_set_path)
processed_test_set = pd.read_csv(processed_test_set_path)

# get the column indices
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [processed_training_set.columns.get_loc(c) for c in col_names]

def main():
    # user defined function for feature engineering
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self  # nothing else to do
        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household,
                             bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    # Pipeline for numerical features

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    # Overall pipeline
    num_attributes = list(processed_training_set.drop("ocean_proximity", axis=1))
    cat_attributes = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", OneHotEncoder(), cat_attributes)
    ])

    #Transfer both training and test set
    strat_train_set_features_prepared = full_pipeline.fit_transform(processed_training_set)
    strat_test_set_features_prepared = full_pipeline.transform(processed_test_set)

    # export to csv
    np.savetxt(prepared_training_set_path, strat_train_set_features_prepared, delimiter=",")
    np.savetxt(prepared_test_set_path, strat_test_set_features_prepared, delimiter=",")

if __name__ == '__main__':
    main()