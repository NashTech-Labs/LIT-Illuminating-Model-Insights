# MLRun and LIT Integration 

## Introduction
We have leveraged Mlrun,an open source platform to orchestrate a pipeline on Sentiment analysis of Flipkart review dataset.This pipeline utilizes machine learning to analyze user reviews and ratings, predicting whether a review is positive or negative. 
The process involves data ingestion, model training, and deployment as a serverless function. Through this pipeline LIT-NLP will fetch the model from the artifact registry of mlrun and will make visualise the model.

## What is MLRun?

MLRun is an open-source MLOps platform designed to streamline the development, deployment, and management of machine learning models. It offers a unified environment for data scientists and engineers to collaborate, automate, and scale their machine learning workflows.

## What is LIT?
Language interpretability tool is an open source data visualization platform made by Google for visualzaing NLP models and offers a wide variety of metrics to test our model. We've been integerating LIT to our pipeline flow to show the plots of our pretrained model

## Getting Started

### Prerequisites

- First we have to set up 2 environments one for mlrun(compatible with only python:3.9) and second for LIT(for more elaborate display of features we used python:3.10).This is done so that our end-end flow of pipeline isn't hindered.
- Install MLRun and LIT following their official documentation.

### Installation

1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/amanknoldus/mlrun_template/Mlrun-Pipeline
   ```
3. Navigate to the project directory.
4. Install the required dependencies using `pip install -r requirements.txt`.
   ```bash
      pip install -r requirements.txt
   ```

## Usage and Results from each component of pipeline

### Data Preparation and Visualization

Before running the pipeline, ensure your dataset is in a format compatible with LIT. This may involve preprocessing or converting your data into a suitable format.For this the Mlrun pipeline is triggered which encodes our dataset to 
numerical format and displays visualization of data.Data preprocessing gives us visualization graphs of our dataset including .describe function, correlation matrix.

![Screenshot from 2024-04-05 14-16-25](https://github.com/amanknoldus/mlrun_template/assets/56683451/ae84b1c4-9869-4464-ac28-e11ade51416c)


![Screenshot from 2024-04-05 14-16-57](https://github.com/amanknoldus/mlrun_template/assets/56683451/ba42036a-465e-4f6d-a4fc-5f2cba0d03ea)


![Screenshot from 2024-04-05 14-19-16](https://github.com/amanknoldus/mlrun_template/assets/56683451/a6921ac6-41e9-45ad-9822-9f344e5661e1)


### Model training
Our transformer model is being trained on our dataset and the logs are displayed on mlrun ui along with the metrics. 

![Screenshot from 2024-04-05 14-29-41](https://github.com/amanknoldus/mlrun_template/assets/56683451/0db4e47e-40fb-40e3-b93e-8aac15f826cf)


![Screenshot from 2024-04-05 14-29-56](https://github.com/amanknoldus/mlrun_template/assets/56683451/3c13898c-bdf0-49a8-8abd-2ee879f30480)





### Serving Model

![Screenshot from 2024-04-05 14-34-45](https://github.com/amanknoldus/mlrun_template/assets/56683451/4caa6885-7756-43e9-b440-95a940f1fdaa)
Our pretrained transformer is successfully deployed to nuclio and therefore our end-end pipeline is completed.




### Running the Pipeline

To execute the MLRun and LIT integration pipeline, run the following command:

```bash
python3 main.py
```

Change environment to python 3:10 

```bash
python3 lit_app.py
```

### LIT-NLP interface
![Screenshot from 2024-04-05 14-45-33](https://github.com/amanknoldus/mlrun_template/assets/56683451/f06b24b0-bff1-4b4d-b368-b820a25c44f0)

Our model is now being visualised effectively on LIT platform and other metrics are also being displayed.


This script will initiate the data ingestion, model training, and serving process.

