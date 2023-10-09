import logging
import os
import pandas as pd
from transformers import AutoTokenizer

from format_shp import PreferencePrompt, sample_shp_data_from_hf, RESPONSE_TOKEN_1, RESPONSE_TOKEN_2


class SHPTransformation(object):
    def __init__(
            self, name, output_dir='data/', train_size=1.0,
            tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        ):
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
        self.tokenizer = tokenizer
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def format_prompt(self, example):
        prompt = PreferencePrompt("", example["human_ref_A"], example["human_ref_B"])
        slack = self.tokenizer.model_max_length - len(self.tokenizer(str(prompt)).input_ids)

        if slack > 0:
            sentences = []
            for s in self.segmenter.segment(PreferencePrompt.clean_text(row["history"])):
                slack -= len(self.tokenizer(s).input_ids)

                if slack > 0:
                    sentences.append(s)

            prompt.post = "".join(sentences)
        return prompt

    def transformation(self, example):
        raise NotImplementedError

    def inverse_transformation(self, example):
        raise NotImplementedError

    def get_output_fn(self, inverse=False):
        name = f'{self.name}_inverse' if inverse else self.name
        train_fn = os.path.join(self.output_dir, f'shp_train_{name}' + (f'_{self.train_size}' if self.train_size < 1.0 else '') + '.csv')
        test_fn = os.path.join(self.output_dir, f'shp_test_{name}.csv')
        return train_fn, test_fn

    def transform(self, inverse=False):
        logging.info(f'Applying {self.name} to SHP')

        if self.train_size < 1:
            train_data = self.train_data.train_test_split(train_size=self.train_size)['train']
        else:
            train_data = self.train_data

        transform_func = self.inverse_transformation if inverse else self.transformation
        train_fn, test_fn = self.get_output_fn(inverse)
        train_data.map(transform_func).to_csv(train_fn)
        self.test_data.map(transform_func).to_csv(test_fn)


class SHPStandardTransformation(SHPTransformation):
    def __init__(self, output_dir, train_size=1.0):
        super().__init__('std', output_dir, train_size)

    def transformation(self, example):
        example['sentence1'] = self.format_prompt(example)
        example['label'] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2

        return {'sentence1': example['sentence1'], 'label': example['label']}


class SHPNullTransformation(SHPTransformation):
    def __init__(self, output_dir, train_size=1.0):
        super().__init__('null', output_dir, train_size)

    def transformation(self, example):
        example['sentence1'] = " "
        example['label'] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2

        return {'sentence1': example['sentence1'], 'label': example['label']}


class SHPWordLengthTransformation(SHPTransformation):
    def __init__(self, output_dir, train_size=1.0):
        super().__init__('word_length', output_dir, train_size)

    def transformation(self, example):
        example['sentence1'] = len(example['human_ref_A'].split()) - len(example['human_ref_B'].split())
        example['label'] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2

        return {'sentence1': example['sentence1'], 'label': example['label']}

    def inverse_transformation(self, example):
        # repeat the shorter sentence to be of the same length as the longer sentence
        A_tokens = example['human_ref_A'].split()
        B_tokens = example['human_ref_B'].split()
        if len(A_tokens) < len(B_tokens):
            example['human_ref_A'] = " ".join(A_tokens * (len(B_tokens) // len(A_tokens)))
        elif len(B_tokens) < len(A_tokens):
            example['human_ref_B'] = " ".join(B_tokens * (len(A_tokens) // len(B_tokens)))

        example['sentence1'] = self.format_prompt(example)
        example['label'] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2

        return {'sentence1': example['sentence1'], 'label': example['label']}


class SHPRawOverlapTransformation(SHPTransformation):
    def __init__(self, output_dir, train_size=1.0):
        super().__init__('raw_overlap', output_dir, train_size)

    def transformation(self, example):
        A_tokens = self.tokenizer.tokenize(example['human_ref_A'])
        B_tokens = self.tokenizer.tokenize(example['human_ref_B'])
        overlap = set(A_tokens) & (set(B_tokens))
        human_ref_A = " ".join([(t if t in overlap else self.tokenizer.mask_token) for t in A_tokens])
        human_ref_B = " ".join([(t if t in overlap else self.tokenizer.mask_token) for t in B_tokens])
        example['sentence1'] = PreferencePrompt("", human_ref_A, human_ref_B)

        example['label'] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2

        return {'sentence1': example['sentence1'], 'label': example['label']}

    def inverse_transformation(self, example):
        A_tokens = self.tokenizer.tokenize(example['human_ref_A'])
        B_tokens = self.tokenizer.tokenize(example['human_ref_B'])
        overlap = set(A_tokens) & (set(B_tokens))
        human_ref_A = " ".join([(t if t not in overlap else self.tokenizer.mask_token) for t in A_tokens])
        human_ref_B = " ".join([(t if t not in overlap else self.tokenizer.mask_token) for t in B_tokens])
        example['sentence1'] = PreferencePrompt("", human_ref_A, human_ref_B)

        example['label'] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2

        return {'sentence1': example['sentence1'], 'label': example['label']}

