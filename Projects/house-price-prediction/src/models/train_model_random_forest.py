from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
import joblib

def main(max_features=6, n_estimators=100):

    #loading data for training
    PREPARED_DATA_PATH = '../../data/processed/project_house_price_prediction'

    prepared_training_set_path = os.path.join(PREPARED_DATA_PATH, 'strat_train_set_features_prepared.csv')
    prepared_training_label_path = os.path.join(PREPARED_DATA_PATH, 'strat_train_set_label.csv')

    prepared_training_set = pd.read_csv(prepared_training_set_path, header=None)

    # Convert to Series
    prepared_training_set_lable = pd.read_csv(prepared_training_label_path).squeeze(axis=1)

    rf_reg = RandomForestRegressor(max_features=max_features, n_estimators=n_estimators)
    rf_reg.fit(prepared_training_set, prepared_training_set_lable)
    print("training completed with parameter:" + " max_feature " + str(max_features) + " n_estimators " + str(n_estimators))
    model_name = 'random_forest.sav'
    joblib.dump(rf_reg, model_name)
    print('model: ' + model_name + 'is now saved')

if __name__ == '__main__':
    main()