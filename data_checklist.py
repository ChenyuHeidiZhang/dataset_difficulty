import logging
import os
from typing import List
import pandas as pd
from transformers import AutoTokenizer

from train import train
from v_info import v_entropy, v_info
from augment_shp import *
from augment import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# def feasibility_check(model_name, data_dir):
#     """
#     Attribute att is feasibile if V_info(att(X) -> Y) > 0.
#     That is, it is possible to learn something using the given attribute.
#     """

#     # compute conditional V-info
#     print(model_name, data_name, experiment)
#     v_info_data = v_info(f"data/{data_name}_{experiment}.csv",
#             f"models/{model_name}_{data_name}_{experiment}", 
#             f"data/{data_name}_null.csv",
#             f"models/{model_name}_{data_name}_null",
#             tokenizer, out_fn=f"PVI/{model_name}_{data_name}_{experiment}.csv")

#     mean_pvi = v_info_data['PVI'].mean()
#     print(f"Mean PVI: {mean_pvi}")



"""
Feasibility Test:
Attribute att is feasibile if V_info(att(X) -> Y) > 0.
That is, it is possible to learn something using the given attribute.

Exclusivity Test:
Attribute att is exclusive if V_info(att'(X) -> Y) = 0.
That is, all V-usable information exists only in the given attribute.
"""

CHECK_TYPE_TO_DATA_TYPES = {
    'feasibility': ['null', 'att'],
    'exclusivity': ['null', 'att_inv'],
    'sufficiency': ['att', 'std'],
    'necessity': ['att_inv', 'std'],
    'regular_vinfo': ['null', 'std'],
}


def data_check(
    data_fn_l, model_l, data_fn_r, model_r, tokenizer, out_fn, input_key='sentence1'
):
    data = pd.read_csv(data_fn)
    data['H_yb'], _, _ = v_entropy(data_fn_l, model_l, tokenizer, input_key=input_key) 
    data['H_yx'], data['correct_yx'], data['predicted_label'] = v_entropy(
        data_fn_r, model_r, tokenizer, input_key=input_key)
    data['PVI'] = data['H_yb'] - data['H_yx']
    if out_fn:
        data.to_csv(out_fn)
    return data



def main(
    transforms,  # dict of transforms, keys must be one or more of 'null', 'att', 'std'
    check_types: List[str],
    train_size=1.0,
    data_dir='data/',
    model_name_or_path='meta-llama/Llama-2-7b-hf',
    model_output_dir='/scr-ssd/chenyuz/llama2-models/',
):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logging.info('Data directory does not exist, created it.')

    data_fns_and_models = {}  # maps data type to list of resources [train_data_fn, test_data_fn, model]
    # data type can be 'null', 'att', 'att_inv', or 'std'

    for check_type in check_types:
        assert check_type in CHECK_TYPE_TO_DATA_TYPES, f"Check type {check_type} not supported"
        for data_type in CHECK_TYPE_TO_DATA_TYPES[check_type]:
            data_fns_and_models[data_type] = []

    # compute data transforms
    logging.info('Computing data transforms...')
    print('Computing data transforms!')
    for data_type in data_fns_and_models.keys():
        logging.info('Computing data transforms for %s' % data_type)
        if data_type == 'att_inv':
            try:
                transform_obj = transforms['att'](data_dir, train_size)
                train_fn, test_fn = transform_obj.get_output_fn(inverse=True)
                if not os.path.exists(train_fn) or not os.path.exists(test_fn):
                    transform_obj.transform(inverse=True)
            except:
                raise NotImplementedError('Inverse attribute transformation not implemented')
        else:
            transform_obj = transforms[data_type](data_dir, train_size)
            train_fn, test_fn = transform_obj.get_output_fn()
            if not os.path.exists(train_fn) or not os.path.exists(test_fn):
                transform_obj.transform()
        data_fns_and_models[data_type].extend([train_fn, test_fn])

    # train models (TODO: sample a small set of data?)
    logging.info('Training models...')
    for data_type in data_fns_and_models.keys():
        train_data_fn, test_data_fn = data_fns_and_models[data_type]
        model_dir_name = train_data_fn.split('/')[-1].split('.')[0]
        model_dir = os.path.join(model_output_dir, model_dir_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logging.info('Model directory does not exist, created it.')

        if os.listdir(model_dir):
            logging.info('Model for %s already exists' % data_type)
            continue

        logging.info('Training model for %s' % data_type)
        train(
            model_name_or_path,
            train_data_fn,
            test_data_fn,
            output_dir=model_dir
        )
        data_fns_and_models[data_type].append(model_dir)

    # compute v-infos
    logging.info('Computing v-infos...')
    for test_name in check_types:
        logging.info('Computing v-infos for %s' % test_name)
        requested_data_types = CHECK_TYPE_TO_DATA_TYPES[test]

        transform_name = transform['att'].name if test_name != 'regular_vinfo' else ''
        out_fn=f"PVI/{test_name}_{transform_name}_{transform['att'].data_name}_{model_name}.csv"

        data_fn_l, model_name_l = data_fns_and_models[requested_data_types[0]]
        data_fn_r, model_name_r = data_fns_and_models[requested_data_types[1]]
        data = data_check(data_fn_l, model_name_l, data_fn_r, model_name_r, tokenizer, out_fn)

        mean_pvi = data['PVI'].mean()
        print(f"{test_name} test, mean PVI: {mean_pvi}")
        if test_name in ['feasibility', 'necessity']:
            criterion = mean_pvi > 0
        elif test_name in ['exclusivity', 'sufficiency']:
            criterion = abs(mean_pvi - 0) < 0.01
        else:
            return

        if criterion:
            print('Test passes.')
        else:
            print('Test fails.')


if __name__ == '__main__':
    main(
        # transforms={'null': SHPNullTransformation, 'att': SHPWordLengthTransformation, 'std': SHPTransformation},
        transforms={'null': DWMWNullTransformation, 'att': DWMWVocabTransformation, 'std': DWMWStandardTransformation},
        check_types=['feasibility'],
        train_size=1.0,  # only 1.0 is supported for DWMW
        data_dir='checklist_data/',
    )

