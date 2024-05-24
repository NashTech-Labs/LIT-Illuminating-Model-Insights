from pathlib import Path


curr_path = Path(__file__).parents[1]
project_path = Path(__file__).parents[2]
data_directory = curr_path / 'data'
model_download_directory = project_path / "models"
flipkart_dataset_path = data_directory / "review_flipkart.csv"
data_for_lit_path = data_directory / "review_flipkarts.csv"
transformed_dataset_path = data_directory / "transformed_data.csv"
model_path = project_path / "huggingface.pkl"
hugging_face = project_path / "src/serving_model/serving.py"
true_model_path = project_path / "models"
file_path = curr_path / 'data_prep/data.py'


project_name = "flipkart-review1"
PACKAGE_MODULE = "transformers"
SERIALIZABLE_TYPES= [dict, list, tuple, str, int, float]
additional_hyperparameters={
            "TRAIN_output_dir": "finetuning-sentiment-model-3000-samples",
            "TRAIN_learning_rate": 2e-5,
            "TRAIN_per_device_train_batch_size": 16,
            "TRAIN_per_device_eval_batch_size": 16,
            "TRAIN_num_train_epochs": 4,
            "TRAIN_weight_decay": 0.01,
            "TRAIN_push_to_hub": False,
            "TRAIN_evaluation_strategy": "epoch",
            "TRAIN_eval_steps": 1,
            "TRAIN_logging_steps": 1,
            "CLASS_num_labels": 3,
            "ignore_mismatched_sizes": True,
            "optim": "adafactor",
        }

tokenizer_file='config.json'
model_file='huggingface.pkl'
serving_container_requirements=['transformers==4.21.3', 'tensorflow==2.9.2', "torch==2.2.2","Datasets==2.10.1"]
model_serving_name='serving-model'
serving_container_image='mlrun/ml-models:1.5.0-rc9'
label_name='sentiment_code'
training_df_path='akshatmehta98/Flipkart-Dataset'
training_component_path='akshatmehta98/roberta-base-fine-tuned-flipkart-reviews-am'
NUM_SAMPLES=100
col_dropping=["product_name","product_price","Rate","Review","labels"]
task_name='sentiment-analysis'
serving_path='/v2/models/flipkart_review_model'
independent_labels=['product_price','Rate']
dependendent_labels=['sentiment_code','Sentiment']
dataset_artifact_path='artifacts/random_dataset.parquet'
basic_training_params= {"hf_dataset": training_df_path,
                "drop_columns": col_dropping,
                "pretrained_tokenizer":training_component_path,
                "pretrained_model":training_component_path,
                "model_class": "transformers.AutoModelForSequenceClassification",
                "label_name": label_name,
                "num_of_train_samples": NUM_SAMPLES,
                "metrics": ["accuracy"],
                "random_state": 42,
                }
