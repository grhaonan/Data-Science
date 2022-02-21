from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import joblib
import os

def main():

    #loading data for training
    PREPARED_DATA_PATH = '../../data/processed/project_house_price_prediction'

    prepared_test_set_path = os.path.join(PREPARED_DATA_PATH, 'strat_test_set_features_prepared.csv')
    prepared_test_label_path = os.path.join(PREPARED_DATA_PATH, 'strat_test_set_label.csv')

    prepared_test_set = pd.read_csv(prepared_test_set_path, header=None)
    # Convert to Series
    prepared_test_set_label = pd.read_csv(prepared_test_label_path).squeeze(axis=1)

    #Loading model
    model_path = './random_forest.sav'
    loaded_model = joblib.load(model_path)
    prediction = loaded_model.predict(prepared_test_set)
    final_mse = mean_squared_error(prepared_test_set_label, prediction)
    final_rmse = np.sqrt(final_mse)

    print('The final rmse is: ' + str(final_rmse))

if __name__ == '__main__':
    main()