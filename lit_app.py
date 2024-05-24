from src.custom_lit_.custom_lit_model import FlipkartModel
from src.custom_lit_.custom_lit_dataset import FlipkartData
from collections.abc import Sequence
import sys
from typing import Optional
from absl import app
from absl import flags
from absl import logging
from src.utils.constant import data_for_lit_path, model_download_directory
from lit_nlp import dev_server
from lit_nlp import server_flags

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODEL_PATH = flags.DEFINE_string(
    "flipkart",
    f"{model_download_directory}",
    "Path to saved model (from transformers library).",
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", 100, "Maximum number of examples to load into LIT. ")


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
    """Returns a LitApp instance for consumption by gunicorn."""
    FLAGS.set_default("server_type", "external")
    FLAGS.set_default("demo_mode", True)
    # Parse flags without calling app.run(main), to avoid conflict with
    # gunicorn command line flags.
    unused = flags.FLAGS(sys.argv, known_only=True)
    if unused:
        logging.info(
            "toxicity_demo:get_wsgi_app() called with unused args: %s", unused
        )
    return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    model_path = _MODEL_PATH.value
    logging.info("Working directory: %s", model_path)

    # Load our trained model.
    datasets = {"flipkart_dataset": FlipkartData(f"{data_for_lit_path}")}
    models = {"flipkart_review": FlipkartModel(model_path)}

    # Truncate datasets if --max_examples is set.
    for name in datasets:
        logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))
        datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
        logging.info("  truncated to %d examples", len(datasets[name]))

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)

