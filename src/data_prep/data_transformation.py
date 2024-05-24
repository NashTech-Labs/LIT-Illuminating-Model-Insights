import pandas as pd
from src.utils.constant import flipkart_dataset_path, transformed_dataset_path
from src.utils.helpers import load_dataset,convert_to_numeric,convert_to_categorical
from datasets import Dataset
from src.utils.constant import independent_labels,dependendent_labels
import logging


class DataPreprocessing():
    @staticmethod
    def data_preprocessor():
        """
        Performs data preprocessing by dropping nan values and by performing
        appropriate feature engineering techniques
        """
        df = load_dataset(flipkart_dataset_path)
        flipkart_dataset = df
        flipkart_dataset = flipkart_dataset.dropna()
        flipkart_dataset=convert_to_numeric(flipkart_dataset,independent_labels)
        flipkart_dataset=convert_to_categorical(flipkart_dataset,dependendent_labels[0],dependendent_labels[1])
        pd.DataFrame.iteritems = pd.DataFrame.items
        flipkart_dataset.to_csv(transformed_dataset_path, index=False)


