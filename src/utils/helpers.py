import pandas as pd
import mlrun
from src.utils.constant import project_name
import zipfile
import logging

logging.basicConfig(level=logging.INFO)


def load_dataset(data_path):
    """
       Use Pandas read_csv function to read the dataset from the specified file path.
       Return the loaded dataset.
       """
    try:
        data_set = pd.read_csv(data_path)
    except FileNotFoundError as f1:
        f1.strerror = 'File not found at the required location'
        raise f1
    except Exception as e1:
        raise Exception(f'Found an exception:%s\n:{e1}')
    finally:
        return data_set


def convert_to_numeric(df, cols):
    """
    Converting the labels to numeric
    """
    try:
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except KeyError as e:
        raise KeyError(f'Specific column not found:{e}')
    except Exception as e:
        raise Exception(f'Unknown Exception occured:{e}')


def convert_to_categorical(df, new_col, label_col):
    """
    Converting the numeric labels to categorical labels
    """
    try:
        df[new_col] = pd.Categorical(df[label_col]).codes
        df[new_col] = df[new_col].astype('Int64')
        return df
    except KeyError as e:
        raise KeyError(f'Specific column not found:{e}')
    except Exception as e:
        raise Exception(f'Unknown exception occured:{e}')


def load_project():
    """
    Loading the project

    :raise HttpStatusError for Mlrun if found
    :raise Exception if unknown exception occurs
    """
    try:
        return mlrun.get_or_create_project(f"{project_name}", context="./", user_project=True)
    except mlrun.errors.MLRunHTTPStatusError as error1:
        error1.strerror = 'Connection to port not found:'
        raise error1


def extract_zip(src_zip_file, dest_dir):
    """
    Helper function for exctraction of zip files.
    """
    try:
        with zipfile.ZipFile(src_zip_file, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        logging.info("Extraction successful!")
    except Exception as e:
        logging.info(f"Error extracting zip file: {e}")
        raise e


def logging_configuration():
    """
    Basic logging configuration
    """
    logging.basicConfig(level=logging.INFO)


def drop_column(df, col):
    """
    Dropping columns from a dataframe
    :df:Dataframe(pd.Dataframe()) format
    :col:Dataframe column in
    """
    df = df.drop(col, axis=1)
    return df
