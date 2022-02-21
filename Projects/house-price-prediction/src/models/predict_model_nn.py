from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import joblib
import os
# Pytorch NN model
import torch
import torch.nn as nn
import torch.nn.functional as nn_f
from torch.autograd import Variable
from sklearn.utils import shuffle
from numpy import genfromtxt

def main():

    #loading data for training
    PREPARED_DATA_PATH = '../../data/processed/project_house_price_prediction'

    prepared_test_set_path = os.path.join(PREPARED_DATA_PATH, 'strat_test_set_features_prepared.csv')
    prepared_test_label_path = os.path.join(PREPARED_DATA_PATH, 'strat_test_set_label.csv')

    # prepared_test_set = pd.read_csv(prepared_test_set_path, header=None)
    prepared_test_set = genfromtxt(prepared_test_set_path, delimiter=',')

    # Convert to Series
    prepared_test_set_label = pd.read_csv(prepared_test_label_path).squeeze(axis=1)

    n_output = 1
    dropout_p = 0
    n_epochs = 300
    learning_rate = 0.01
    batch_size = 50
    n_batches = 330
    n_feature = 16
    size_hidden = 50

    class Net(torch.nn.Module):
        def __init__(self, n_feature, size_hidden, n_output, dropout_p):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, size_hidden)
            self.predict = torch.nn.Linear(size_hidden, n_output)
            self.dropout = nn.Dropout(dropout_p)

        def forward(self, x):
            x = nn_f.relu(self.hidden(x))
            x = self.dropout(self.predict(x))
            return x

    loaded_model = Net(n_feature, size_hidden, n_output, dropout_p)

    #Loading model
    model_path = './pytorch_nn.pth'
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    X_test_set = Variable(torch.FloatTensor(prepared_test_set))
    nn_predict_result_tensor = loaded_model(X_test_set)
    nn_predict_result = nn_predict_result_tensor.data[:, 0].numpy()

    final_mse = mean_squared_error(prepared_test_set_label, nn_predict_result)
    final_rmse = np.sqrt(final_mse)
    print('The final rmse is: ' + str(final_rmse))

if __name__ == '__main__':
    main()