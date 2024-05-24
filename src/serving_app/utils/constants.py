from pathlib import Path
project_path = Path(__file__).parents[3]
project_name = "flipkart-review1"
serving_model_path = project_path / "models"
tokenizer_file="config.json"
path_file='/home/nashtech/Desktop/Mlrun-final-pipeline/src/serving_app/run_app.py'
training_component_path='akshatmehta98/roberta-base-fine-tuned-flipkart-reviews-am'
