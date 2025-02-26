import os
from collections.abc import Iterable, Sequence
import re
import threading
from typing import Any, Optional
from lit_nlp.api import types
import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import model_utils
from lit_nlp.lib import file_cache
from lit_nlp.lib import utils
import numpy as np
import tensorflow as tf
import transformers

JsonDict = lit_types.JsonDict
Spec = lit_types.Spec
TFSequenceClassifierOutput = (
    transformers.modeling_tf_outputs.TFSequenceClassifierOutput
)


@attr.s(auto_attribs=True, kw_only=True)
class GlueModelConfig(object):
    """Config options for a GlueModel."""
    # Preprocessing options
    max_seq_length: int = 128
    inference_batch_size: int = 32
    # Input options
    text_a_name: str = "sentence1"
    text_b_name: Optional[str] = "sentence2"  # set to None for single-segment
    label_name: str = "label"
    # Output options
    labels: Optional[list[str]] = None  # set to None for regression
    null_label_idx: Optional[int] = None
    compute_grads: bool = True  # if True, compute and return gradients.
    output_attention: bool = True
    output_embeddings: bool = True

    @classmethod
    def init_spec(cls) -> lit_types.Spec:
        return {
            "model_name_or_path": lit_types.String(
                default="bert-base-uncased",
                required=False,
            ),
            "max_seq_length": lit_types.Integer(
                default=128,
                max_val=512,
                min_val=1,
                required=False,
            ),
            "inference_batch_size": lit_types.Integer(
                default=32,
                max_val=64,
                min_val=1,
                required=False,
            ),
            "compute_grads": lit_types.Boolean(default=True, required=False),
            "output_attention": lit_types.Boolean(default=True, required=False),
            "output_embeddings": lit_types.Boolean(default=True, required=False),
        }


class GlueModel(lit_model.BatchedModel):
    """

      This is a general-purpose classification or regression model. It works for
      one- or two-segment input, and predicts either a multiclass label or
      a regression score. See GlueModelConfig for available options.

      This implements the LIT API for inference (e.g. input_spec(), output_spec(),
      and predict()), but also provides a train() method to run fine-tuning.

      This is a full-featured implementation, which includes embeddings, attention,
      gradients, as well as support for the different input and output types above.
      For a more minimal example, see ../simple_tf2_demo.py.
      """

    def _verify_num_layers(self, hidden_states: Sequence[Any]):
        """Verify correct # of layer activations returned."""
        # First entry is embeddings, then output from each transformer layer.
        expected_hidden_states_len = self.model.config.num_hidden_layers + 1
        actual_hidden_states_len = len(hidden_states)
        if actual_hidden_states_len != expected_hidden_states_len:
            raise ValueError(
                "Unexpected size of hidden_states. Should be one "
                "more than the number of hidden layers to account "
                "for the embeddings. Expected "
                f"{expected_hidden_states_len}, got "
                f"{actual_hidden_states_len}."
            )

    @property
    def is_regression(self) -> bool:
        return self.config.labels is None

    def __init__(self,
                 model_name_or_path="bert-base-uncased",
                 **config_kw):
        self.config = GlueModelConfig(**config_kw)
        self._load_model(model_name_or_path)
        self._lock = threading.Lock()

    def _load_model(self, model_name_or_path):
        """Load model. Can be overridden for testing."""
        # Normally path is a directory; if it's an archive file, download and
        # extract to the transformers cache.
        if model_name_or_path.endswith(".tar.gz"):
            model_name_or_path = file_cache.cached_path(
                model_name_or_path, extract_compressed_file=True
            )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path)
        self.vocab = self.tokenizer.convert_ids_to_tokens(
            range(len(self.tokenizer)))
        model_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=1 if self.is_regression else len(self.config.labels),
            return_dict=False,  # default for training; overridden for predict
            output_attentions=self.config.output_attention,
        )
        self.model = model_utils.load_pretrained(
            transformers.TFAutoModelForSequenceClassification,
            model_name_or_path,
            config=model_config)

    def _get_tokens(self, ex: JsonDict, field_name: str) -> list[str]:
        with self._lock:
            return (ex.get("tokens_" + field_name) or
                    self.tokenizer.tokenize(ex[field_name]))

    def _preprocess(self, inputs: Iterable[JsonDict]) -> dict[str, tf.Tensor]:
        # Use pretokenized input if available.
        tokens_a = [self._get_tokens(ex, self.config.text_a_name) for ex in inputs]
        tokens_b = None
        if self.config.text_b_name:
            tokens_b = [
                self._get_tokens(ex, self.config.text_b_name) for ex in inputs
            ]
        # Use custom tokenizer call to make sure we don't mangle pre-split
        # wordpieces in pretokenized input.
        encoded_input = model_utils.batch_encode_pretokenized(
            self.tokenizer,
            tokens_a,
            tokens_b,
            max_length=self.config.max_seq_length)
        return encoded_input  # pytype: disable=bad-return-type

    def _make_dataset(self, inputs: Iterable[JsonDict]) -> tf.data.Dataset:
        """Make a tf.data.Dataset from inputs in LIT format."""
        encoded_input = self._preprocess(inputs)
        if self.is_regression:
            labels = tf.constant([ex[self.config.label_name] for ex in inputs],
                                 dtype=tf.float32)
        else:
            labels = tf.constant([
                self.config.labels.index(ex[self.config.label_name]) for ex in inputs
            ],
                dtype=tf.int64)
        # encoded_input is actually a transformers.BatchEncoding
        # object, which tf.data.Dataset doesn't like. Convert to a regular dict.
        return tf.data.Dataset.from_tensor_slices((dict(encoded_input), labels))

    def save(self, path: str):
        """Save model weights and tokenizer info.

    To re-load, pass the path to the constructor instead of the name of a
    base model.

    Args:
      path: directory to save to. Will write several files here.
    """
        if not os.path.isdir(path):
            os.mkdir(path)
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    def _segment_slicers(self, tokens: list[str]):
        """Slicers along the tokens dimension for each segment.

    For tokens ['[CLS]', a0, a1, ..., '[SEP]', b0, b1, ..., '[SEP]'],
    we want to get the slices [a0, a1, ...] and [b0, b1, ...]

    Args:
      tokens: <string>[num_tokens], including special tokens

    Returns:
      (slicer_a, slicer_b), slice objects
    """
        try:
            split_point = tokens.index(self.tokenizer.sep_token)
        except ValueError:
            split_point = len(tokens) - 1
        slicer_a = slice(1, split_point)  # start after [CLS]
        slicer_b = slice(split_point + 1, len(tokens) - 1)  # end before last [SEP]
        return slicer_a, slicer_b

    def _postprocess(self, output: dict[str, Any]):
        """Per-example postprocessing, on NumPy output."""
        ntok = output.pop("ntok")
        output["tokens"] = self.tokenizer.convert_ids_to_tokens(
            output.pop("input_ids")[:ntok])

        # Tokens for each segment, individually.
        slicer_a, slicer_b = self._segment_slicers(output["tokens"])
        output["tokens_" + self.config.text_a_name] = output["tokens"][slicer_a]
        if self.config.text_b_name:
            output["tokens_" + self.config.text_b_name] = output["tokens"][slicer_b]

        # Embeddings for each segment, individually.
        if self.config.output_embeddings:
            output["input_embs_" + self.config.text_a_name] = (
                output["input_embs"][slicer_a])
            if self.config.text_b_name:
                output["input_embs_" + self.config.text_b_name] = (
                    output["input_embs"][slicer_b])

        # Gradients for each segment, individually.
        if self.config.compute_grads:
            # Gradients for the CLS token.
            output["cls_grad"] = output["input_emb_grad"][0]
            output["token_grad_" +
                   self.config.text_a_name] = output["input_emb_grad"][slicer_a]
            if self.config.text_b_name:
                output["token_grad_" +
                       self.config.text_b_name] = output["input_emb_grad"][slicer_b]
            # is updated.
            if not self.is_regression:
                # Return the label corresponding to the class index used for gradients.
                output[self.config.label_name] = self.config.labels[
                    output[self.config.label_name]
                ]  # pytype: disable=container-type-mismatch

            # Remove "input_emb_grad" since it's not in the output spec.
            del output["input_emb_grad"]

        if not self.config.output_attention:
            return output

        # Process attention.
        for key in output:
            if not re.match(r"layer_(\d+)/attention", key):
                continue
            output[key] = output[key][:, :ntok, :ntok].transpose((0, 2, 1))
            output[key] = output[key].copy()  # pytype: disable=attribute-error

        return output

    def _scatter_embs(self, passed_input_embs, input_embs, batch_indices,
                      offsets):
        """Scatters custom passed embeddings into the default model embeddings.

    Args:
      passed_input_embs: <tf.float32>[num_scatter_tokens], the custom passed
        embeddings to be scattered into the default model embeddings.
      input_embs: the default model embeddings.
      batch_indices: the indices of the embeddings to replace in the format
        (batch_index, sequence_index).
      offsets: the offset from which to scatter the custom embedding (number of
        tokens from the start of the sequence).

    Returns:
      The default model embeddings with scattered custom embeddings.
    """

        # <float32>[scatter_batch_size, num_tokens, emb_size]
        filtered_embs = [emb for emb in passed_input_embs if emb is not None]

        # Prepares update values that should be scattered in, i.e. one for each
        # of the (scatter_batch_size * num_tokens) word embeddings.
        # <np.float32>[scatter_batch_size * num_tokens, emb_size]
        updates = np.concatenate(filtered_embs)

        # Prepares indices in format (batch_index, sequence_index) for all
        # values that should be scattered in, i.e. one for each of the
        # (scatter_batch_size * num_tokens) word embeddings.
        scatter_indices = []
        for (batch_index, sentence_embs, offset) in zip(batch_indices,
                                                        filtered_embs, offsets):
            for (token_index, _) in enumerate(sentence_embs):
                scatter_indices.append([batch_index, token_index + offset])

        # Scatters passed word embeddings into embeddings gathered from tokens.
        # <tf.float32>[batch_size, num_tokens + num_special_tokens, emb_size]
        return tf.tensor_scatter_nd_update(input_embs, scatter_indices, updates)

    def scatter_all_embeddings(self, inputs, input_embs):
        """Scatters custom passed embeddings for text segment inputs.

    Args:
      inputs: the model inputs, which contain any custom embeddings to scatter.
      input_embs: the default model embeddings.

    Returns:
      The default model embeddings with scattered custom embeddings.
    """
        # Gets batch indices of any word embeddings that were passed for text_a.
        passed_input_embs_a = [ex.get("input_embs_" + self.config.text_a_name)
                               for ex in inputs]
        batch_indices_a = [index for (index, emb) in enumerate(
            passed_input_embs_a) if emb is not None]

        # If word embeddings were passed in for text_a, scatter them into the
        # embeddings, gathered from the input ids. 1 is passed in as the offset
        # for each, since text_a starts at index 1, after the [CLS] token.
        if batch_indices_a:
            input_embs = self._scatter_embs(
                passed_input_embs_a, input_embs, batch_indices_a,
                offsets=np.ones(len(batch_indices_a), dtype=np.int64))

        if self.config.text_b_name:
            # Gets batch indices of any word embeddings that were passed for text_b.
            passed_input_embs_b = [ex.get("input_embs_" + self.config.text_b_name)
                                   for ex in inputs]
            batch_indices_b = [
                index for (index, emb) in enumerate(passed_input_embs_b)
                if emb is not None]
            if batch_indices_b:
                lengths = np.array([len(embed) for embed in passed_input_embs_a
                                    if embed is not None])
                input_embs = self._scatter_embs(
                    passed_input_embs_b, input_embs, batch_indices_b,
                    offsets=(lengths + 2))
        return input_embs

    def get_target_scores(self, inputs: Iterable[JsonDict], scores):
        """Get target-class scores, as a 1D tensor.

    Args:
      inputs: list of input examples
      scores: <tf.float32>[batch_size, num_classes], either logits or probas

    Returns:
      <tf.float32>[batch_size] target scores for each input
    """
        arg_max = tf.math.argmax(scores, axis=-1).numpy()
        grad_classes = [
            ex.get(self.config.label_name, arg_max[i])
            for (i, ex) in enumerate(inputs)
        ]
        # Convert the class names to indices if needed.
        grad_idxs = [
            self.config.labels.index(label) if isinstance(label, str) else label
            for label in grad_classes
        ]
        # list of tuples (batch idx, label idx)
        gather_indices = list(enumerate(grad_idxs))
        # <tf.float32>[batch_size]
        return tf.gather_nd(scores, gather_indices), grad_idxs

    ##
    # LIT API implementation
    def max_minibatch_size(self):
        return self.config.inference_batch_size

    def get_embedding_table(self):
        if hasattr(self.model.roberta.embeddings, "word_embeddings"):
            return self.vocab, self.model.roberta.embeddings.word_embeddings.numpy()
        else:
            return self.vocab, self.model.roberta.embeddings.weight.numpy()

    def predict_minibatch(self, inputs: Iterable[JsonDict]):
        # Use watch_accessed_variables to save memory by having the tape do nothing
        # if we don't need gradients.
        with tf.GradientTape(
                watch_accessed_variables=self.config.compute_grads) as tape:
            encoded_input = self._preprocess(inputs)

            # Gathers word embeddings from model embedding layer using input ids
            # of the tokens.
            input_ids = encoded_input["input_ids"]
            if hasattr(self.model.roberta.embeddings, "word_embeddings"):
                word_embeddings = self.model.roberta.embeddings.word_embeddings
            else:
                word_embeddings = self.model.roberta.embeddings.weight
            # <tf.float32>[batch_size, num_tokens, emb_size]
            input_embs = tf.gather(word_embeddings, input_ids)

            # Scatter in any passed in embeddings.
            # <tf.float32>[batch_size, num_tokens, emb_size]
            input_embs = self.scatter_all_embeddings(inputs, input_embs)

            tape.watch(input_embs)  # Watch input_embs for gradient calculation.

            model_inputs = encoded_input.copy()
            model_inputs["input_ids"] = None
            out: TFSequenceClassifierOutput = self.model(
                model_inputs,
                inputs_embeds=input_embs,
                training=False,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True)

            batched_outputs = {
                "input_ids": encoded_input["input_ids"],
                "ntok": tf.reduce_sum(encoded_input["attention_mask"], axis=1),
                "cls_emb": out.hidden_states[-1][:, 0],  # last layer, first token
            }

            if self.config.output_embeddings:
                batched_outputs["input_embs"] = input_embs

                self._verify_num_layers(out.hidden_states)

                # <float32>[batch_size, num_tokens, 1]
                token_mask = tf.expand_dims(
                    tf.cast(encoded_input["attention_mask"], tf.float32), axis=2)
                # <float32>[batch_size, 1]
                denom = tf.reduce_sum(token_mask, axis=1)
                for i, layer_output in enumerate(out.hidden_states):
                    # layer_output is <float32>[batch_size, num_tokens, emb_dim]
                    # average over tokens to get <float32>[batch_size, emb_dim]
                    batched_outputs[f"layer_{i}/avg_emb"] = tf.reduce_sum(
                        layer_output * token_mask, axis=1) / denom

            if self.config.output_attention:
                if len(out.attentions) != self.model.config.num_hidden_layers:
                    raise ValueError("Unexpected size of attentions. Should be the same "
                                     "size as the number of hidden layers. Expected "
                                     f"{self.model.config.num_hidden_layers}, got "
                                     f"{len(out.attentions)}.")
                for i, layer_attention in enumerate(out.attentions):
                    batched_outputs[f"layer_{i + 1}/attention"] = layer_attention

            if self.is_regression:
                # <tf.float32>[batch_size]
                batched_outputs["score"] = tf.squeeze(out.logits, axis=-1)
                # <tf.float32>[batch_size], a single target per example
                scalar_targets = batched_outputs["score"]
            else:
                # <tf.float32>[batch_size, num_labels]
                batched_outputs["probas"] = tf.nn.softmax(out.logits, axis=-1)
                # <tf.float32>[batch_size], a single target per example
                scalar_targets, grad_idxs = self.get_target_scores(
                    inputs, batched_outputs["probas"]
                )
                if self.config.compute_grads:
                    batched_outputs[self.config.label_name] = tf.convert_to_tensor(
                        grad_idxs
                    )

        if self.config.compute_grads:
            # <tf.float32>[batch_size, num_tokens, emb_dim]
            batched_outputs["input_emb_grad"] = tape.gradient(
                scalar_targets, input_embs
            )

        detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
        # Sequence of dicts, one per example.
        unbatched_outputs = utils.unbatch_preds(detached_outputs)
        return map(self._postprocess, unbatched_outputs)

    def input_spec(self) -> Spec:
        ret = {}
        ret[self.config.text_a_name] = lit_types.TextSegment()
        ret["tokens_" + self.config.text_a_name] = lit_types.Tokens(
            parent=self.config.text_a_name, required=False)

        if self.config.text_b_name:
            ret[self.config.text_b_name] = lit_types.TextSegment()
            ret["tokens_" + self.config.text_b_name] = lit_types.Tokens(
                parent=self.config.text_b_name, required=False)

        if self.is_regression:
            ret[self.config.label_name] = lit_types.Scalar(required=False)
        else:
            ret[self.config.label_name] = lit_types.CategoryLabel(
                required=False, vocab=self.config.labels)

        if self.config.output_embeddings:
            # The input_embs_ fields are used for Integrated Gradients.
            text_a_embs = "input_embs_" + self.config.text_a_name
            ret[text_a_embs] = lit_types.TokenEmbeddings(
                align="tokens", required=False)
            if self.config.text_b_name:
                text_b_embs = "input_embs_" + self.config.text_b_name
                ret[text_b_embs] = lit_types.TokenEmbeddings(
                    align="tokens", required=False
                )
        return ret

    def output_spec(self) -> Spec:
        ret = {"tokens": lit_types.Tokens()}
        ret["tokens_" + self.config.text_a_name] = lit_types.Tokens(
            parent=self.config.text_a_name)
        if self.config.text_b_name:
            ret["tokens_" + self.config.text_b_name] = lit_types.Tokens(
                parent=self.config.text_b_name)
        if self.is_regression:
            ret["score"] = lit_types.RegressionScore(parent=self.config.label_name)
        else:
            ret["probas"] = lit_types.MulticlassPreds(
                parent=self.config.label_name,
                vocab=self.config.labels,
                null_idx=self.config.null_label_idx)

        if self.config.output_embeddings:
            ret["cls_emb"] = lit_types.Embeddings()
            # Average embeddings, one per layer including embeddings.
            for i in range(1 + self.model.config.num_hidden_layers):
                ret[f"layer_{i}/avg_emb"] = lit_types.Embeddings()

            # The input_embs_ fields are used for Integrated Gradients.
            ret["input_embs_" + self.config.text_a_name] = lit_types.TokenEmbeddings(
                align="tokens_" + self.config.text_a_name)
            if self.config.text_b_name:
                text_b_embs = "input_embs_" + self.config.text_b_name
                ret[text_b_embs] = lit_types.TokenEmbeddings(align="tokens_" +
                                                                   self.config.text_b_name)

        # Gradients, if requested.
        if self.config.compute_grads:
            ret["cls_grad"] = lit_types.Gradients(
                align=("score" if self.is_regression else "probas"),
                grad_for="cls_emb",
                grad_target_field_key=self.config.label_name,
            )
            if not self.is_regression:
                ret[self.config.label_name] = lit_types.CategoryLabel(
                    required=False, vocab=self.config.labels
                )
            if self.config.output_embeddings:
                text_a_token_grads = "token_grad_" + self.config.text_a_name
                ret[text_a_token_grads] = lit_types.TokenGradients(
                    align="tokens_" + self.config.text_a_name,
                    grad_for="input_embs_" + self.config.text_a_name,
                    grad_target_field_key=self.config.label_name,
                )
                if self.config.text_b_name:
                    text_b_token_grads = "token_grad_" + self.config.text_b_name
                    ret[text_b_token_grads] = lit_types.TokenGradients(
                        align="tokens_" + self.config.text_b_name,
                        grad_for="input_embs_" + self.config.text_b_name,
                        grad_target_field_key=self.config.label_name,
                    )

        if self.config.output_attention:
            # Attention heads, one field for each layer.
            for i in range(self.model.config.num_hidden_layers):
                ret[f"layer_{i + 1}/attention"] = lit_types.AttentionHeads(
                    align_in="tokens", align_out="tokens")
        return ret


class FlipkartModel(GlueModel):
    """Classification model on MultiNLI."""

    def __init__(self, *args, **kw):
        super().__init__(
            *args,
            text_a_name="sentence",
            text_b_name=None,
            labels=['0', '1', '2'],
            **kw)

    def input_spec(self) -> types.Spec:
        """Describe the inputs to the model."""
        return {
            'sentence': lit_types.String()
        }


