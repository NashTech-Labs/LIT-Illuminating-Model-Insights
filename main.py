import time
import os
from src.model.training_component_transformer import ModelTraining
from src.model.fetch_model import LoadModel
from src.data_prep.run_component import DataVisualization
from src.serving_model.model_serving import ServingModel
from src.utils.helpers import logging_configuration


if __name__ == '__main__':
    """For running visualizations """
    logging_configuration()
    data_visual_inst = DataVisualization()
    data_visual_inst.runner()
    time.sleep(5)
    # """For Model traning on ui"""
    model_train_inst = ModelTraining()
    model_train_inst.trainer_runner()
    time.sleep(10)
    """For download Model for LIT"""
    load_model_inst = LoadModel()
    load_model_inst.load_model_mlrun_artifact()
    # time.sleep(50)
    """For model serving"""
    os.system("python3 src/serving_app/server.py")
