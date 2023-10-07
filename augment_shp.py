import logging
import os
import pandas as pd
from transformers import AutoTokenizer

from format_shp import PreferencePrompt, sample_shp_data_from_hf, RESPONSE_TOKEN_1, RESPONSE_TOKEN_2


class SHPTransformation(object):
    def __init__(self, name='full', output_dir='data/', train_size=1.0):
        """
        Args:
            name: Transformation name
            output_dir: where to save the CSV with the transformed attribute
            train_size: fraction of the training data to use
        """
        self.train_data, self.test_data = sample_shp_data_from_hf(2.0, 5)
        self.data_name = 'shp'
        self.name = name
        self.output_dir = output_dir
        self.train_size = train_size

    def transformation(self, example):
        prompt = PreferencePrompt("", example["human_ref_A"], example["human_ref_B"])
        example['sentence1'] = prompt
        # target = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2
        # example['label'] = target

        return example

    def inverse_transformation(self, example):
        raise NotImplementedError

    def get_output_fn(self, inverse=False, train=True):
        name = f'{self.name}_inverse' if inverse else self.name
        if train:
            return os.path.join(self.output_dir, f'shp_train_{name}' + (f'_{self.train_size}' if self.train_size < 1.0 else '') + '.csv')
        else:
            return os.path.join(self.output_dir, f'shp_test_{name}.csv')

    def transform(self, inverse=False):
        logging.info(f'Applying {self.name} to SHP')

        if self.train_size < 1:
            train_data = self.train_data.train_test_split(train_size=self.train_size)['train']
        else:
            train_data = self.train_data

        transform_func = self.inverse_transformation if inverse else self.transformation

        train_data.map(transform_func).to_csv(self.get_output_fn(inverse, train=True))
        self.test_data.map(transform_func).to_csv(self.get_output_fn(inverse, train=False))


class SHPNullTransformation(SHPTransformation):
    def __init__(self, output_dir, train_size=1.0):
        super().__init__('null', output_dir, train_size)

    def transformation(self, example):
        example['sentence1'] = " "
        return example


class SHPWordLengthTransformation(SHPTransformation):
    def __init__(self, output_dir, train_size=1.0):
        super().__init__('word_length', output_dir, train_size)

    def transformation(self, example):
        example['sentence1'] = len(example['human_ref_A'].split()) - len(example['human_ref_B'].split())
        return example


class SHPRawOverlapTransformation(SHPTransformation):
    def __init__(self, output_dir, train_size=1.0):
        super().__init__('raw_overlap', output_dir, train_size)
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf') 

    def transformation(self, example):
        A_tokens = self.tokenizer.tokenize(example['human_ref_A'])
        B_tokens = self.tokenizer.tokenize(example['human_ref_B'])
        overlap = set(A_tokens) & (set(B_tokens))
        human_ref_A = " ".join([(t if t in overlap else self.tokenizer.mask_token) for t in A_tokens])
        human_ref_B = " ".join([(t if t in overlap else self.tokenizer.mask_token) for t in B_tokens])
        example['sentence1'] = PreferencePrompt("", human_ref_A, human_ref_B)
        return example

    def inverse_transformation(self, example):
        A_tokens = self.tokenizer.tokenize(example['human_ref_A'])
        B_tokens = self.tokenizer.tokenize(example['human_ref_B'])
        overlap = set(A_tokens) & (set(B_tokens))
        human_ref_A = " ".join([(t if t not in overlap else self.tokenizer.mask_token) for t in A_tokens])
        human_ref_B = " ".join([(t if t not in overlap else self.tokenizer.mask_token) for t in B_tokens])
        example['sentence1'] = PreferencePrompt("", human_ref_A, human_ref_B)
        return example
