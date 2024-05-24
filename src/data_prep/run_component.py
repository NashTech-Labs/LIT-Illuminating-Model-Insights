import mlrun
from src.data_prep.data_transformation import DataPreprocessing
from src.utils.constant import file_path, project_name
from src.utils.helpers import load_project,logging_configuration
import os
import logging
from mlrun.errors import MLRunHTTPStatusError as error1


class DataVisualization:
    def __init__(self):
        logging.info('Preprocessing component....')
    def runner(self):
        """
               Executes the data visualization process.

               Raises:
                   error1: If there's an MLRun HTTP status error.
                   Exception: If any other unexpected error occurs.
        """
        try:
            project = load_project()
            preprocessor_inst = DataPreprocessing()
            preprocessor_inst.data_preprocessor()
            data_gen_fn = project.set_function(func=f"{file_path}", name="flipkart-transformed", kind="job",
                                                    image="mlrun/mlrun",
                                                    handler="flipkart")
            project.save()
            gen_data_run = project.run_function("flipkart-transformed", params={"format": "csv"}, local=True)

            describe_func = mlrun.import_function("hub://describe")

            describe_run = describe_func.run(
                name="task-describe",
                handler='analyze',
                inputs={"table": os.path.abspath("artifacts/random_dataset.parquet")},
                params={"name": "flipkart dataset", "label_column": "Sentiment"},
                local=True
            )

        except ConnectionRefusedError as c1:
            raise error1

        except Exception as e:
            raise e