import mlrun
from datasets import load_metric
import logging
from src.utils.constant import project_name, additional_hyperparameters, training_df_path, training_component_path, \
    label_name, NUM_SAMPLES, col_dropping, basic_training_params
from src.utils.helpers import load_project
from src.utils.helpers import logging_configuration
import pickle


class ModelTraining:
    def __init__(self):
        logging.info('Reached training component....')

    @staticmethod
    def trainer_runner():
        """
              Executes the model training process using the Hugging Face Trainer.

              Raises:
                  Exception: If any unexpected error occurs during the training process.
        """
        try:
            logging.info('Loading model training component.....')
            project = load_project()
            hugging_face_classifier_trainer = mlrun.import_function("hub://hugging_face_classifier_trainer")
            additional_parameters = additional_hyperparameters
            train_run = hugging_face_classifier_trainer.run(params={
                **basic_training_params,
                **additional_parameters
            },
                handler="train",
                local=True,
            )
        except Exception as e:
            raise Exception('Raised an issue:\n%s', e)
