# GPT-2 Epicurious Fine-Tuning (TensorFlow)
# SciNet DAT112: Neural Network Programming
# Adapted by Layal Jbara 
#
# This script fine-tunes a HuggingFace GPT-2 transformer
# on the Epicurious recipes dataset.
#######################################################################

"""
Fine_Tune.py

This script fine-tunes a HuggingFace GPT-2 model on the Epicurious
recipes dataset using TensorFlow/Keras.

Main tasks:

1. Detect GPU(s) and set memory growth.
2. Load raw text splits (train/test/validation) with 🤗 Datasets.
3. Initialize tokenizer and model config from a pretrained GPT-2.
4. Tokenize text, then group tokens into fixed-length blocks.
5. Convert to tf.data datasets with appropriate collator.
6. Train (fit) the model with validation and save a checkpoint.

Reference implementation:
https://github.com/huggingface/transformers/blob/main/examples/tensorflow/language-modeling/run_clm.py
"""

#######################################################################
# Imports
#######################################################################

import os
from itertools import chain

from datasets import load_dataset

import tensorflow as tf
import tensorflow.keras.callbacks as kc

from transformers import (
    CONFIG_NAME,
    TF2_WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
    DefaultDataCollator,
    TFAutoModelForCausalLM
)

#######################################################################
# (Optional) GPU visibility override
#######################################################################

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#######################################################################
# GPU configuration (SciNet / Mist cluster)
#######################################################################

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

#######################################################################
# Main
#######################################################################

def main():
    ###################################################################
    # Hyperparameters & argument-style dicts
    ###################################################################
    MODEL_NAME = 'gpt2'
    batch_size = 32

    training_args = {
        'output_dir': 'output',
        'overwrite_output_dir': False,
        'per_device_train_batch_size': batch_size,
        'num_train_epochs': 250
    }

    # NOTE: In this script, `block_size` is reused later after adjustment.
    data_args = {
        'preprocessing_num_workers': 4,
        'overwrite_cache': False,
        'block_size': batch_size
    }

    model_args = {'model_name_or_path': MODEL_NAME}

    ###################################################################
    # Check for an existing checkpoint
    ###################################################################
    print('Checking checkpoint directory.')
    checkpoint = None
    if ((len(os.listdir(training_args['output_dir'])) > 0) and
        (not training_args['overwrite_output_dir'])):
        config_path = training_args['output_dir'] + '/' + CONFIG_NAME
        weights_path = training_args['output_dir'] + '/' + TF2_WEIGHTS_NAME
        if os.path.isfile(config_path) and os.path.isfile(weights_path):
            checkpoint = training_args['output_dir']

    ###################################################################
    # Load datasets (raw text files)
    ###################################################################
    print('Grabbing data.')
    data_files = {
        'train': 'epicurious.recipes.training',
        'test': 'epicurious.recipes.testing',
        'validation': 'epicurious.recipes.validation'
    }
    raw_datasets = load_dataset('text', data_files=data_files)

    ###################################################################
    # Load tokenizer and config
    ###################################################################
    print('Grabbing tokenizer and config.')
    tokenizer = AutoTokenizer.from_pretrained(model_args['model_name_or_path'])
    config = AutoConfig.from_pretrained(model_args['model_name_or_path'])

    ###################################################################
    # Tokenization
    ###################################################################
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # The batching technique below strips newlines; we add them back.
    def tokenize_function(examples):
        return tokenizer(
            [example + "\n" for example in examples[text_column_name]],
            max_length=512,
            truncation=True
        )

    print('Tokenizing.')
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args['preprocessing_num_workers'],
        remove_columns=column_names,
        load_from_cache_file=not data_args['overwrite_cache'],
        desc="Running tokenizer on dataset",
    )

    ###################################################################
    # Group tokens into fixed-length blocks
    ###################################################################
    block_size = min(data_args['block_size'], tokenizer.model_max_length)

    # Concatenate and split into chunks of `block_size`.
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k]))
                                 for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print(f'Blocking the data (block_size={block_size}).')
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args['preprocessing_num_workers'],
        load_from_cache_file=not data_args['overwrite_cache'],
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset  = lm_datasets["validation"]

    ###################################################################
    # Initialize / load model
    ###################################################################
    if checkpoint is not None:
        model = TFAutoModelForCausalLM.from_pretrained(checkpoint, config=config)
    elif model_args['model_name_or_path']:
        model = TFAutoModelForCausalLM.from_pretrained(
            model_args['model_name_or_path'], config=config
        )

    model.resize_token_embeddings(len(tokenizer))

    ###################################################################
    # Build tf.data pipelines
    ###################################################################
    data_collator = DefaultDataCollator(return_tensors="tf")
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )

    print('Converting data to TensorFlow datasets.')
    tf_train_dataset = train_dataset.to_tf_dataset(
        # Labels are part of the inputs; model uses internal loss.
        columns=[col for col in train_dataset.features if col != "special_tokens_mask"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
        drop_remainder=True,
    ).with_options(options)

    tf_eval_dataset = eval_dataset.to_tf_dataset(
        columns=[col for col in eval_dataset.features if col != "special_tokens_mask"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
        drop_remainder=True,
    ).with_options(options)

    ###################################################################
    # Compile & train
    ###################################################################
    batches_per_epoch = len(train_dataset) // batch_size
    model.compile(optimizer='adam')

    history = model.fit(
        tf_train_dataset,
        validation_data=tf_eval_dataset,
        epochs=int(training_args['num_train_epochs']),
        steps_per_epoch=len(train_dataset) // training_args['per_device_train_batch_size'],
        verbose=2
    )

    ###################################################################
    # Save checkpoint
    ###################################################################
    if training_args['output_dir'] is not None:
        model.save_pretrained(training_args['output_dir'])


#######################################################################
# Entrypoint
#######################################################################

if __name__ == "__main__":
    main()