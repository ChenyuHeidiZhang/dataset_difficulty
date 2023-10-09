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
    data_fn_l, model_l, data_fn_r, model_r, model_name_or_path,
    out_fn, input_key='sentence1', use_lora=True
):
    data = pd.read_csv(data_fn_r)
    tokenizer = model_name_or_path

    if use_lora:
        data['H_yb'], _, _ = v_entropy(
            data_fn_l, model_name_or_path, tokenizer, input_key=input_key, lora_model_path=model_l) 
        data['H_yx'], data['correct_yx'], data['predicted_label'] = v_entropy(
            data_fn_r, model_name_or_path, tokenizer, input_key=input_key, lora_model_path=model_r)
    else:
        data['H_yb'], _, _ = v_entropy(
            data_fn_l, model_l, tokenizer, input_key=input_key) 
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
    use_lora=True,
):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logging.info('Data directory does not exist, created it.')

    data_fns_and_models = {}  # maps data type to list of resources [train_data_fn, test_data_fn, model]
    # data type can be 'null', 'att', 'att_inv', or 'std'
    # model can be a directory or the actual model

    for check_type in check_types:
        assert check_type in CHECK_TYPE_TO_DATA_TYPES, f"Check type {check_type} not supported"
        for data_type in CHECK_TYPE_TO_DATA_TYPES[check_type]:
            data_fns_and_models[data_type] = []

    # compute / find data transforms
    logging.info('Looking for existing data transform files...')
    for data_type in data_fns_and_models.keys():
        if data_type == 'att_inv':
            try:
                transform_obj = transforms['att'](data_dir, train_size)
                train_fn, test_fn = transform_obj.get_output_fn(inverse=True)
                if not os.path.exists(train_fn) or not os.path.exists(test_fn):
                    logging.info('Computing data transforms for %s' % data_type)
                    transform_obj.transform(inverse=True)
            except:
                raise NotImplementedError('Inverse attribute transformation not implemented')
        else:
            transform_obj = transforms[data_type](data_dir, train_size)
            train_fn, test_fn = transform_obj.get_output_fn()
            if not os.path.exists(train_fn) or not os.path.exists(test_fn):
                logging.info('Computing data transforms for %s' % data_type)
                transform_obj.transform()
        data_fns_and_models[data_type].extend([train_fn, test_fn])

    # train / find models
    logging.info('Trying to find models...')
    for data_type in data_fns_and_models.keys():
        train_data_fn, test_data_fn = data_fns_and_models[data_type]
        model_dir_name = train_data_fn.split('/')[-1].split('.')[0]
        model_dir = os.path.join(model_output_dir, model_dir_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logging.info('Model directory does not exist, created it.')
        data_fns_and_models[data_type].append(model_dir)

        if os.listdir(model_dir):
            logging.info('Model for %s already exists' % data_type)
            continue

        logging.info('Training model for %s' % data_type)
        _, _ = train(
            model_name_or_path,
            train_data_fn,
            test_data_fn,
            output_dir=model_dir
        )

    # compute v-infos
    logging.info('Computing v-infos...')
    for test_name in check_types:
        logging.info('Computing v-infos for %s' % test_name)
        requested_data_types = CHECK_TYPE_TO_DATA_TYPES[test_name]

        _, data_fn_l, model_l = data_fns_and_models[requested_data_types[0]]
        _, data_fn_r, model_r = data_fns_and_models[requested_data_types[1]]
        transform_name = transforms['att'].name if test_name != 'regular_vinfo' else ''
        model_name = model_name_or_path.split('/')[-1]
        out_fn=f"checklist_PVI/{test_name}_{transforms['att'].data_name}_{transform_name}_{model_name}.csv"

        data = data_check(
            data_fn_l, model_l, data_fn_r, model_r, model_name_or_path, out_fn,
            use_lora=use_lora)

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
        check_types=['necessity', 'exclusivity', 'sufficiency'],
        train_size=1.0,  # only 1.0 is supported for DWMW
        data_dir='checklist_data/',
    )

