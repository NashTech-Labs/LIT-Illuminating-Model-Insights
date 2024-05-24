from mlrun.artifacts import get_model
import zipfile
import logging
from src.utils.constant import model_download_directory, project_name
from src.utils.helpers import load_project,extract_zip
import mlrun


logging.basicConfig(level=logging.INFO)



class LoadModel():

    @staticmethod
    def load_model_mlrun_artifact():
        """
        Loading the model from mlrun artifact,followed by unzipping
        and placing the models and tokenizers in their folders.
        """
        project = load_project()
        models = project.list_models()
        model_path = None
        for model in models:
            logging.info('Model:\n%s',model)
            model_path = model.uri
        model_obj, model_file, extra_data = get_model(model_path)
        token_object = extra_data['tokenizer']
        dest_dir = model_download_directory
        src_zip_file = str(token_object)
        extract_zip(src_zip_file, dest_dir)
        src_zip_file = model_obj
        extract_zip(src_zip_file, dest_dir)