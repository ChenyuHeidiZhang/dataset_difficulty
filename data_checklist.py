import logging
import os
from typing import List
import pandas as pd
from transformers import AutoTokenizer

from v_info import v_entropy, v_info


def feasibility_check(model_name, data_dir):
    """
    Attribute att is feasibile if V_info(att(X) -> Y) > 0.
    That is, it is possible to learn something using the given attribute.
    """

    # compute conditional V-info
    print(model_name, data_name, experiment)
    v_info_data = v_info(f"data/{data_name}_{experiment}.csv",
            f"models/{model_name}_{data_name}_{experiment}", 
            f"data/{data_name}_null.csv",
            f"models/{model_name}_{data_name}_null",
            tokenizer, out_fn=f"PVI/{model_name}_{data_name}_{experiment}.csv")

    mean_pvi = v_info_data['PVI'].mean()
    print(f"Mean PVI: {mean_pvi}")


def exclusivity_check(att_transform, data_dir):
    """
    """
    pass

def sufficiency_check():
    pass


def necessity_check():
    pass


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
    'sufficiency': ['att', 'full'],
    'necessity': ['att_inv', 'full'],
    'regular_vinfo': ['null', 'full'],
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
    transforms,  # dict of transforms, keys must be one or more of 'null', 'att', 'full'
    check_types: List[str],
    train_size=1.0,
    data_dir='data/'
):

    data_fns_and_models = {}  # maps data type to [data fn, model]
    # data type can be 'null', 'att', 'att_inv', or 'full'

    for check_type in check_types:
        assert check_type in CHECK_TYPE_TO_DATA_TYPES, f"Check type {check_type} not supported"
        for data_type in CHECK_TYPE_TO_DATA_TYPES[check_type]:
            data_fns_and_models[data_type] = []

    # compute data transforms
    for data_type in data_fns_and_models.keys():
        if data_type == 'att_inv':
            try:
                transforms['att'](data_dir, train_size).transform(inverse=True)
                data_fns_and_models[data_type].append(
                    transforms['att'].get_output_fn(inverse=True, train=True))
            except:
                raise NotImplementedError('Inverse attribute transformation not implemented')
        else:
            transforms[data_type](data_dir, train_size).transform()
            data_fns_and_models[data_type].append(
                transforms[data_type].get_output_fn(train=True))

    # train model (TODO: sample a small set of data?)



    # compute v-infos

    for test_name in check_types:
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
    main()
