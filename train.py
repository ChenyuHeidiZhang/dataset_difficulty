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
    default_data_collator,
    get_scheduler,
    set_seed,
)
from peft import (
    get_peft_model,
    LoraConfig,
)

logger = logging.getLogger(__name__)

# currently supported tasks: dwmw, shp


def check_input_args(train_file, validation_file, output_dir):
    assert train_file is not None
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
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    pad_to_max_length=False,
    max_length=4096,
    learning_rate=5e-5,
    num_train_epochs=1,
    max_train_steps=None,
    num_warmup_steps=0,
    gradient_accumulation_steps=1,
    enable_fsdp=False,  # not supported yet
    use_lora=True,
    output_dir='/nlp/scr/chenyuz',
    sentence1_key='sentence1',
    sentence2_key=None,
):
    '''Trains LLaMA model for a classification task.
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
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists.
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column.

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
    logging.info('num_labels: %d' % num_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, pad_token_id=tokenizer.eos_token_id)
    else:
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=config,
    )

    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    if use_lora:
        peft_config = LoraConfig(
            task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    padding = "max_length" if pad_to_max_length else False
    max_length = min(max_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        result["labels"] = [label_to_id[l] for l in examples["label"]]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if pad_to_max_length:
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size)

    # Optimizer
    optimizer = AdamW(params=model.parameters(), lr=learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    metric = load_metric("accuracy")

    # Train!
    total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

        torch.cuda.empty_cache() # free memory
        model.eval()

        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")


    if output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)


    return model, tokenizer
