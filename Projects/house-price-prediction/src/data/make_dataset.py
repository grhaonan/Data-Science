# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
#from dotenv import find_dotenv, load_dotenv


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

# Defien file path
RAW_DATA_PATH = '../../data/raw/project_house_price_prediction'
FILE_NAME = 'housing.csv'
PROCESSED_DATA_PATH = '../../data/processed/project_house_price_prediction'

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    # Read data
    raw_data_path = os.path.join(RAW_DATA_PATH, FILE_NAME)

    raw_data_pd = pd.read_csv(raw_data_path)

    # Categorise the mediuam_income feature and apply stratified sampling
    raw_data_pd["income_cat"] = pd.cut(raw_data_pd["median_income"], bins = [0.,1.5,3.0,4.5,6.,np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=35)

    for train_index, test_index in split.split(raw_data_pd, raw_data_pd["income_cat"]):
        strat_train_set = raw_data_pd.loc[train_index]
        strat_test_set = raw_data_pd.loc[test_index]


    # remove 'income_cat' column from both original and splitted dataset
    for each_set in (strat_train_set, strat_test_set):
        each_set.drop("income_cat", axis=1, inplace=True)

    raw_data_pd.drop("income_cat", axis=1, inplace=True)

    # split attributes and label

    strat_train_set_features = strat_train_set.drop("median_house_value", axis =1)
    strat_train_set_label = strat_train_set["median_house_value"].copy()

    strat_test_set_features = strat_test_set.drop("median_house_value", axis =1)
    strat_test_set_label = strat_test_set["median_house_value"].copy()

    processed_training_set_path = os.path.join(PROCESSED_DATA_PATH, 'strat_train_set_features.csv')
    processed_training_set_label_path = os.path.join(PROCESSED_DATA_PATH, 'strat_train_set_label.csv')

    processed_test_set_path = os.path.join(PROCESSED_DATA_PATH, 'strat_test_set_features.csv')
    processed_test_set_label_path = os.path.join(PROCESSED_DATA_PATH, 'strat_test_set_label.csv')

    strat_train_set_features.to_csv(processed_training_set_path, index=False)
    strat_train_set_label.to_csv(processed_training_set_label_path, index=False)

    strat_test_set_features.to_csv(processed_test_set_path, index=False)
    strat_test_set_label.to_csv(processed_test_set_label_path, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(RAW_DATA_PATH,PROCESSED_DATA_PATH)
