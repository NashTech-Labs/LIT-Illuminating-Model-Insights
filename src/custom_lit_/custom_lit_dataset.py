from typing import Optional

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.lib import file_cache
import pandas as pd
import tensorflow_datasets as tfds


def load_tfds(*args, do_sort=True, **kw):
    ret = list(tfds.as_numpy(tfds.load(*args, download=True, try_gcs=True, **kw)))
    if do_sort:
        ret.sort(key=lambda ex: ex['idx'])
    return ret


class FlipkartData(lit_dataset.Dataset):
    LABELS = ['0', '1', '2']

    def load_from_csv(self, path: str):
        path = file_cache.cached_path(path)
        with open(path) as fd:
            df = pd.read_csv(fd)
            df = df[['sentence', 'label']]
        if set(df.columns) != set(self.spec().keys()):
            raise ValueError(
                f'CSV columns {list(df.columns)} do not match expected'
                f' {list(self.spec().keys())}.'
            )
        df['label'] = df.label.map(str)
        return df.to_dict(orient='records')

    def load_from_tfds(self, split: str):
        if split not in self.TFDS_SPLITS:
            raise ValueError(
                f"Unsupported split '{split}'. Allowed values: {self.TFDS_SPLITS}"
            )
        ret = []
        for ex in load_tfds('glue/sst2', split=split):
            ret.append({
                'sentence': ex['sentence'].decode('utf-8'),
                'label': self.LABELS[ex['label']],
            })
        return ret

    def __init__(
            self, path_or_splitname: str, max_examples: Optional[int] = None
    ):
        if path_or_splitname.endswith('.csv'):
            self._examples = self.load_from_csv(path_or_splitname)[:max_examples]
        else:
            self._examples = self.load_from_tfds(path_or_splitname)[:max_examples]

    @classmethod
    def init_spec(cls) -> lit_types.Spec:
        return {
            'path_or_splitname': lit_types.String(
                default='validation', required=True
            ),
            'max_examples': lit_types.Integer(
                default=1000, min_val=0, max_val=10_000, required=False
            ),
        }

    def spec(self):
        return {
            'sentence': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=self.LABELS)
        }

