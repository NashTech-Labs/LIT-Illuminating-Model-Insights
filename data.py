import pandas as pd
from src.utils.constant import transformed_dataset_path
import mlrun
import os


@mlrun.handler()
def flipkart(context, format="csv"):
    """a function which generates the dataset"""
    dataset = pd.read_csv(transformed_dataset_path)

    reviews_labels = pd.DataFrame(data=dataset.Sentiment, columns=["Sentiment"])
    try:
        os.mkdir('artifacts')
    except:
        pass
    dataset.to_parquet("artifacts/random_dataset.parquet")
    context.logger.info("saving flipkart dataframe")


if __name__ == "__main__":
    with mlrun.get_or_create_ctx(
            "flipkart", upload_artifacts=True
    ) as context:
        flipkart(context, context.get_param("format", "csv"))
