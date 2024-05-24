from src.utils.helpers import load_project,load_dataset,convert_to_categorical,convert_to_numeric,logging_configuration,drop_column,extract_zip
from src.utils.constant import flipkart_dataset_path,independent_labels,dependendent_labels
from src.data_prep.run_component import DataVisualization
from src.model.training_component_transformer import ModelTraining
import pytest
import pandas as pd
import os
import logging
import sys
from src.serving_model.model_serving import ServingModel
logging_configuration()
class Test_Runner():
    @staticmethod
    @pytest.fixture
    def common_dataset():
        df=pd.read_csv(flipkart_dataset_path)
        return df

    def test_drop_col(self,common_dataset):
        """
        test_drop_col: Test function to ensure dropping a single column from the dataset.
        params:
        @common_dataset:Dataset of type pd.Dataframe
        """
        new_df=drop_column(common_dataset,'product_name')
        assert 'product_name' not in new_df

    def test_drop_col2(self,common_dataset):
        """
        Test function to ensure dropping multiple columns from the dataset
        params:
        @common_dataset:Dataset of type pd.Dataframe
        """
        new_df=drop_column(common_dataset,['product_name','Sentiment'])
        assert 'product_name' and 'sentiment' not in new_df

    def test_unknown_drop(self,common_dataset):
        """
        Test function to check the behavior when trying to drop an unknown column.
        params:
        @common_dataset:Dataset of type pd.Dataframe
        """
        with pytest.raises(KeyError) as exc_info:
            new_df = drop_column(common_dataset,['unknown_col'])

        assert str(exc_info.value) == '"[\'unknown_col\'] not found in axis"'


    def test_load_dataset(self):
        """
        Test function to load a dataset
        """
        try:
            load_dataset(flipkart_dataset_path)
        except Exception as e:
            raise Exception(f'Raised unknown exception:{e}')

    def test_convert_numeric(self,common_dataset):
        """
        Test function to convert columns to numeric data type.
        params:
        @common_dataset:Dataframe of type pd.Dataframe
        """
        try:
            convert_to_numeric(common_dataset,independent_labels)
        except Exception as e:
            raise Exception(f'Exception occured:%s{e}')
        except KeyError as k1:
            raise KeyError(f'Column names might not exist{k1}')



    def test_pipeline_flow(self):
        """
        Test function to simulate the pipeline flow by running the main script.
        """
        try:
            os.system('python3 main.py')
        except Exception as e:
            logging.warning('Got an unknown exception at one of the component....')
            raise e

    def test_mlrun_load_project(self):
        """
        Test function to load a project
        """
        try:
            load_project()
        except Exception as e:
            logging.warning('Got unknown exception while loading the project....')
            raise e

    def test_convert_to_categorical(self,common_dataset):
        """
        Test function to convert columns to categorical data type
        params:
        @common_dataset:Dataframe of type pd.Dataframe
        """
        try:
            convert_to_categorical(common_dataset,dependendent_labels[0],dependendent_labels[1])

        except Exception as e:
            raise e


    def test_drop_na(self,common_dataset):
        """
        Test function to drop rows with missing values from the dataset.
        params:
        @common_dataset:Datset of type pd.Dataframe
        """
        try:
            orignal_shape=common_dataset.shape
            new_df=common_dataset.dropna()
            new_shape=new_df.shape
            assert orignal_shape!=new_shape
        except Exception as e:
            raise e





    def test_serving_nuclio(self):
        """
        Test function to ensure the serving model runs successfully
        """
        try:
            serve_inst=ServingModel()
            serve_inst.serving_runner()
        except Exception as e:
            raise e

    def test_extract_valid_zip(self):
        """
        Test function to extract a valid zip file

        """
        src_zip_file = 'model.zip'
        dest_dir = './test_suite'
        extract_zip(src_zip_file, dest_dir)
        assert os.path.exists(dest_dir)
        assert len(os.listdir(dest_dir)) > 0

    def test_extract_invalid_zip(self):
        """
        Test function to handle extraction of an invalid zip file
        """
        src_zip_file = 'invalid_zip_file.zip'
        dest_dir = 'destination_directory'
        with pytest.raises(Exception):  # Assuming extract_zip raises an exception for invalid zip files
            extract_zip(src_zip_file, dest_dir)
        assert not os.path.exists(dest_dir)
