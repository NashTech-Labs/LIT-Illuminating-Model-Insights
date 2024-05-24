import mlrun
import os
import pickle
from src.utils.constant import hugging_face, true_model_path, project_path, model_path, project_name,tokenizer_file,model_file,serving_container_requirements,model_serving_name,serving_container_image,task_name,serving_path
from src.utils.helpers import load_project
from transformers import AutoConfig, AutoModelForSequenceClassification
import logging
from mlrun.errors import MLRunFatalFailureError as error2
from src.utils.helpers import logging_configuration


class ServingModel:
    def __init__(self):
        logging.info('Loading serving model on nucleio component....')
    @staticmethod
    def serving_runner():
        """
        Serving the model through the pipeline
        """
        try:
            project =load_project()
            config = AutoConfig.from_pretrained(f'{true_model_path}/{tokenizer_file}')
            model = AutoModelForSequenceClassification.from_pretrained(f'{true_model_path}/pytorch_model.bin',
                                                                       config=config)
            with open(f'{project_path}/{model_file}', 'wb') as f:
                pickle.dump(model, f)

            serving_fn = mlrun.code_to_function(name=model_serving_name, filename=f'{hugging_face}',
                                                kind='serving', image=serving_container_image,
                                                requirements=serving_container_requirements)

            serving_fn.add_model(
                'flipkart_review_model',
                class_name='HuggingFaceModelServer',
                model_path=f'{model_path}',  # This is not used, just for enabling the process.

                task=task_name,
                model_class="AutoModelForSequenceClassification",
                tokenizer_class="AutoTokenizer",
                tokenizer_name=f'{true_model_path}',
            )
            server = serving_fn.to_mock_server()
            result = server.test(
                path=serving_path,
                body={"inputs": ["good product"]}
            )
            logging.info(f"prediction: {result['outputs']}")
            serving_fn.deploy()

        except ConnectionRefusedError as e:
            raise error2('Connection error found',e)

        except Exception as e:
            raise e
