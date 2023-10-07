import logging
import math
import os
import random
import torch
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)


task_to_keys = {
    "dwmw": ("sentence1", None),
    "shp": ("sentence1", None),
}


def check_input_args(train_file, validation_file, output_dir):
    assert train_file is not None:
    extension = train_file.split(".")[-1]
    assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."

    if validation_file is not None:
        extension = validation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)


def train(
    model_name_or_path,
    train_file,
    validation_file,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=1,
    output_dir='/nlp/scr/chenyuz',
):
    '''Trains model for a classification task.
    '''

    check_input_args(train_file, validation_file, output_dir)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # Loading the dataset from local csv or json file.
    data_files = {}
    data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    extension = train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, pad_token_id=tokenizer.eos_token_id)
    else:
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    sentence1_key, sentence2_key = "sentence1", None

