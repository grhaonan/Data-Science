import os
import pandas as pd
import joblib
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

    prepared_training_set_path = os.path.join(PREPARED_DATA_PATH, 'strat_train_set_features_prepared.csv')
    prepared_training_label_path = os.path.join(PREPARED_DATA_PATH, 'strat_train_set_label.csv')

    #prepared_training_set = pd.read_csv(prepared_training_set_path, header=None)

    prepared_training_set = genfromtxt(prepared_training_set_path, delimiter=',')
    # Convert to Series
    prepared_training_set_lable = pd.read_csv(prepared_training_label_path).squeeze(axis=1)

    # Prepare training hypterparameters
    n_output =1
    dropout_p = 0
    n_epochs = 300
    learning_rate = 0.01
    batch_size = 50
    n_batches = len(prepared_training_set) // batch_size
    n_feature = prepared_training_set.shape[1]
    # rule of thumb, 2/3 of input size plus output layer size
    #size_hidden = int((2*n_feature)/3) + n_output
    size_hidden = 50

    # Create the model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Executing the model on:", device)

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

    net_linear = Net(n_feature, size_hidden, n_output, dropout_p)

    # Define optimiser
    optim = torch.optim.Adam(net_linear.parameters(), lr=learning_rate)

    # Define loss function
    criterion = torch.nn.MSELoss()

    #train model
    running_loss = 0
    running_loss_diagram = []
    for epoch in range(n_epochs):
        X_train, y_train = shuffle(prepared_training_set, prepared_training_set_lable)
        # Mini batch learning
        for i in range(n_batches):
            start_p = i * batch_size
            end_p = start_p + batch_size
            inputs = Variable(torch.FloatTensor(X_train[start_p:end_p]))
            # since y_train is series, we need to convert it to np array by using .values
            labels = Variable(torch.FloatTensor(y_train.values[start_p:end_p]))

            #reset the gradients
            optim.zero_grad()

            #forward
            outputs = net_linear(inputs)
            loss = criterion(outputs, torch.unsqueeze(labels, dim=1))
            loss.backward()
            optim.step()

            #print statistics
            running_loss += loss.item()

        #print('Epoch {}'.format(epoch+1), "loss: ",running_loss)
        running_loss_diagram.append(running_loss)
        running_loss = 0.0

    print("training completed with last loss:" + str(running_loss_diagram[-1]))
    model_save_path = 'pytorch_nn.pth'
    torch.save(net_linear.state_dict(), model_save_path)
    print('model: ' + model_save_path + 'is now saved')

if __name__ == '__main__':
    main()