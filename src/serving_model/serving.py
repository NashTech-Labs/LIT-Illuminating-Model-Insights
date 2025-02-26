import pickle
from abc import ABC
from importlib import import_module
from typing import List

from transformers import pipeline

import mlrun.serving
from src.utils.constant import PACKAGE_MODULE, SERIALIZABLE_TYPES


class HuggingFaceModelServer(mlrun.serving.V2ModelServer, ABC):
    """
    Hugging Face Model serving_model class, inheriting the V2ModelServer class for being initialized automatically by the
    model server and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
    """

    def __init__(
            self,
            context: mlrun.MLClientCtx,
            name: str,
            task: str,
            model_path: str = None,
            model_name: str = None,
            model_class: str = None,
            tokenizer_name: str = None,
            tokenizer_class: str = None,
            framework: str = None,
            **class_args,
    ):
        """
        Initialize a serving_model class for a Hugging face model.

        :param context:         The mlrun context to work with
        :param name:            The name of this server to be initialized
        :param model_path:      Not in use. When adding a model pass any string value
        :param model_name:      The model's name in the Hugging Face hub
                                e.g., `nlptown/bert-base-multilingual-uncased-sentiment`
        :param model_class:     The model's class type object which can be passed as the class's name (string).
                                Must be provided and to be matched with `model_name`.
                                e.g., `AutoModelForSequenceClassification`
        :param tokenizer_name:  The tokenizer's name in the Hugging Face hub
                                e.g., `nlptown/bert-base-multilingual-uncased-sentiment`
        :param tokenizer_class: The model's class type object which can be passed as the class's name (string).
                                Must be provided and to be matched with `model_name`.
                                e.g., `AutoTokenizer`
        :param framework:       The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified
                                framework must be installed.
                                If no framework is specified, will default to the one currently installed.
                                If no framework is specified and both frameworks are installed, will default to the
                                framework of the `model`, or to PyTorch if no model is provided.
        :param class_args:      -
        """
        super(HuggingFaceModelServer, self).__init__(
            context=context,
            name=name,
            model_path=model_path,
            **class_args,
        )
        self.task = task
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.framework = framework
        self.pipe = None

    def load(self):
        """load and initialize the model and/or other elements"""
        if self.model_path is None and self.model_class:
            model_object = getattr(import_module(PACKAGE_MODULE), self.model_class)
            self.model = model_object.from_pretrained(self.model_name)
        if self.tokenizer_class:
            tokenizer_object = getattr(
                import_module(PACKAGE_MODULE), self.tokenizer_class
            )
            self.tokenizer = tokenizer_object.from_pretrained(self.tokenizer_name)
        if self.model_path is not None:
            model_file, extra_data = self.get_model('.pkl')
            self.model = pickle.load(open(model_file, 'rb'))

        self.pipe = pipeline(
            task=self.task,
            model=self.model or self.model_name,
            tokenizer=self.tokenizer,
            framework=self.framework,
        )

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        if self.pipe is None:
            raise ValueError("Please use `.load()`")
        try:
            if isinstance(body["inputs"][0], dict):
                result = [self.pipe(**_input) for _input in body["inputs"]]
            else:
                result = self.pipe(body["inputs"])
            # replace list of lists of dicts into a list of dicts:
            if all(isinstance(res, list) for res in result):
                new_result = [res[0] for res in result]
                result = new_result

            non_serializable_types = []
            for res in result:
                for key, val in res.items():
                    if type(val) not in SERIALIZABLE_TYPES:
                        non_serializable_types.append(str(type(val)))
                        res[key] = str(val)
            if non_serializable_types:
                self.context.logger.info(
                    f"Non-serializable types: {non_serializable_types} were casted to strings"
                )
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
        return result
