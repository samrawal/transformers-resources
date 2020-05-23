# Adapted from `run_glue.py` from HuggingFace Transformers library
# https://github.com/huggingface/transformers
'''Functions to finetune Transformers models for sequence classification, with custom Data Processors .'''


import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

# add custom processors & metrics
import custom_processors
# metrics
from custom_processors import custom_compute_metrics as compute_metrics # override glue_compute_metrics
# processors
for k, v in custom_processors.custom_processors.items():
    processors[k] = v
# output modes
for k, v in custom_processors.custom_output_modes.items():
    output_modes[k] = v

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    #"albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    #"xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

def set_seed(args):
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['n_gpu'] > 0:
        torch.cuda.manual_seed_all(args['seed'])

def load_config(config_path='config.json'):
    config = json.load(open(config_path, 'r'))
    required_params = {'data_dir', 'model_type', 'model_name_or_path', 'task_name', 'output_dir'}
    other_params = {'config_name': '', # Pretrained config name or path if not the same as model_name
                    'tokenizer_name': '', # Pretrained tokenizer name or path if not the same as model_name
                    'cache_dir': '', # Where do you want to store the pretrained models downloaded from s3
                    'max_seq_length': 128, # The max total input seq len after tokenization (longer->truncate; shorter->pad)
                    'do_train': False, # Whether to run training
                    'do_eval': False, # Whether to run eval on the dev set
                    'evaluate_during_training': False, # Run eval during training at each logging step
                    'do_lower_case': True, # Set to True if using uncased model
                    'per_gpu_train_batch_size': 8, # batch size per GPU/CPU for training
                    'per_gpu_eval_batch_size': 8, # batch size per GPU/CPU for evaluation
                    'gradient_accumulation_steps': 1, # Number of udpates steps to accumulate before performing backward/update pass
                    'learning_rate': 5e-5, # Initial learning rate for Adam
                    'weight_decay': 0.0, # Weight decay if we apply some
                    'adam_epsilon': 1e-8, # Epsilon for Adam optimizer
                    'max_grad_norm': 1.0, # Max gradient norm
                    'num_train_epochs': 3.0, # Total number of training epochs to perform
                    'max_steps': -1, # If > 0: Set total num of training steps to perform. Override num_train_epochs
                    'warmup_steps': 0, # Linear warmup over warmup_steps
                    'logging_steps': 50, # Log every X updates steps
                    'save_steps': 50, # Save checkpoint every X updates steps
                    'eval_all_checkpoints': True, # Eval all checkpoints starting w/ same prefix as model_name ending and ending with step number
                    'no_cuda': False, # Avoid using CUDA when available
                    'overwrite_output_dir': True, # Overwrite the content of the output dir
                    'overwrite_cache': True, # Overwrite the cached train and eval sets
                    'seed': 42, # Random seed for initialization
                    'fp16': False, # Whether to use the 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
                    'fpt_opt_level': 'O1', # For fp16: Apex AMP optimization elvel selected in ['O0', 'O1', 'O2', 'O3']
                    'local_rank': -1, # For distributed training: local_rank
                    'server_ip': '', # For distant debugging
                    'server_port': '', # For distant debugging
    }
    args = {}
    for p in required_params:
        args[p] = config[p]
    for p in other_params:
        args[p] = config[p] if p in config else other_params[p]

    return args

def load_tokenizer_model(args):
    if args['local_rank'] not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args['model_type'] = args['model_type'].lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
    config = config_class.from_pretrained(
        args['config_name'] if args['config_name'] else args['model_name_or_path'],
        num_labels=args['num_labels'],
        finetuning_task=args['task_name'],
        cache_dir=args['cache_dir'] if args['cache_dir'] else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args['tokenizer_name'] if args['tokenizer_name'] else args['model_name_or_path'],
        do_lower_case=args['do_lower_case'],
        cache_dir=args['cache_dir'] if args['cache_dir'] else None,
    )
    model = model_class.from_pretrained(
        args['model_name_or_path'],
        from_tf=bool(".ckpt" in args['model_name_or_path']),
        config=config,
        cache_dir=args['cache_dir'] if args['cache_dir'] else None,
    )

    if args['local_rank'] == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args['device'])

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = load_and_cache_examples(args, args['task_name'], tokenizer, evaluate=False)

    return config, tokenizer, model
    
def train(args, data, model, tokenizer):
    """ Train the model """
    if args['local_rank'] in [-1, 0]:
        tb_writer = SummaryWriter()

    args['train_batch_size'] = args['per_gpu_train_batch_size'] * max(1, args['n_gpu'])
    train_sampler = RandomSampler(data) if args['local_rank'] == -1 else DistributedSampler(data)
    train_dataloader = DataLoader(data, sampler=train_sampler, batch_size=args['train_batch_size'])

    if args['max_steps'] > 0:
        t_total = args['max_steps']
        args['num_train_epochs'] = args['max_steps'] // (len(train_dataloader) // args['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args['weight_decay'],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args['model_name_or_path'], "optimizer.pt")) and os.path.isfile(
        os.path.join(args['model_name_or_path'], "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args['model_name_or_path'], "scheduler.pt")))

    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args['local_rank'] != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args['local_rank']], output_device=args['local_rank'], find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(data))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Instantaneous batch size per GPU = %d", args['per_gpu_train_batch_size'])
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args['train_batch_size']
        * args['gradient_accumulation_steps']
        * (torch.distributed.get_world_size() if args['local_rank'] != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args['model_name_or_path']):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args['model_name_or_path'].split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args['gradient_accumulation_steps'])
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args['gradient_accumulation_steps'])

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args['num_train_epochs']), desc="Epoch", disable=args['local_rank'] not in [-1, 0]
    )

    set_seed(args)  # Added here for reproductibility

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args['local_rank'] not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args['device']) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args['model_type'] != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args['model_type'] in ["bert", "xlnet"] else None
                )  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                if args['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args['local_rank'] in [-1, 0] and args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    logs = {}
                    if (
                        args['local_rank'] == -1 and args['evaluate_during_training']
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args['logging_steps']
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args['local_rank'] in [-1, 0] and args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    args_json = args
                    args_json['device'] = str(args_json['device'])
                    json.dump(args_json, open(os.path.join(args['output_dir'], "training_args.json"), 'w'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args['max_steps'] > 0 and global_step > args['max_steps']:
                epoch_iterator.close()
                break
        if args['max_steps'] > 0 and global_step > args['max_steps']:
            train_iterator.close()
            break

    if args['local_rank'] in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args['task_name'] == "mnli" else (args['task_name'],)
    eval_outputs_dirs = (args['output_dir'], args['output_dir'] + "-MM") if args['task_name'] == "mnli" else (args['output_dir'],)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args['local_rank'] in [-1, 0]:
            os.makedirs(eval_output_dir)

        args['eval_batch_size'] = args['per_gpu_eval_batch_size'] * max(1, args['n_gpu'])
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

        # multi-gpu eval
        if args['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args['eval_batch_size'])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args['device']) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args['model_type'] != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args['model_type'] in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args['output_mode'] == "classification":
            preds = np.argmax(preds, axis=1)
        elif args['output_mode'] == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        preds_eval_file = os.path.join(eval_output_dir, prefix, "eval_preds.txt")
        with open(preds_eval_file, "w") as writer:
            for p in preds:
                writer.write('{}\n'.format(p))

    return results

def predict(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args['task_name'] == "mnli" else (args['task_name'],)
    eval_outputs_dirs = (args['output_dir'], args['output_dir'] + "-MM") if args['task_name'] == "mnli" else (args['output_dir'],)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        # if not os.path.exists(eval_output_dir) and args['local_rank'] in [-1, 0]:
            # os.makedirs(eval_output_dir)

        args['eval_batch_size'] = args['per_gpu_eval_batch_size'] * max(1, args['n_gpu'])
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

        # multi-gpu eval
        if args['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args['eval_batch_size'])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args['device']) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args['model_type'] != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args['model_type'] in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args['output_mode'] == "classification":
            preds = np.argmax(preds, axis=1)
        elif args['output_mode'] == "regression":
            preds = np.squeeze(preds)
        #result = compute_metrics(eval_task, preds, out_label_ids)
        #results.update(result)
    return preds#, results

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args['local_rank'] not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args['data_dir'],
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args['model_name_or_path'].split("/"))).pop(),
            str(args['max_seq_length']),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args['overwrite_cache']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args['model_type'] in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args['max_seq_length'],
            output_mode=output_mode,
            pad_on_left=bool(args['model_type'] in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args['model_type'] in ["xlnet"] else 0,
        )
        if args['local_rank'] in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args['local_rank'] == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def glue(config_path='config.json'):
    args = load_config(config_path)
    if (
        os.path.exists(args['output_dir'])
        and os.listdir(args['output_dir'])
        and args['do_train']
        and not args['overwrite_output_dir']
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args['output_dir']
            )
        )

    # Setup distant debugging if needed
    if args['server_ip'] and args['server_port']:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args['server_ip'], args['server_port']), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args['local_rank'] == -1 or args['no_cuda']:
        device = torch.device("cuda" if torch.cuda.is_available() and not args['no_cuda'] else "cpu")
        args['n_gpu'] = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args['local_rank'])
        device = torch.device("cuda", args['local_rank'])
        torch.distributed.init_process_group(backend="nccl")
        args['n_gpu'] = 1
    args['device'] = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args['local_rank'] in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args['local_rank'],
        device,
        args['n_gpu'],
        bool(args['local_rank'] != -1),
        args['fp16'],
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE Task
    args['task_name'] = args['task_name'].lower()
    if args['task_name'] not in processors:
        raise ValueError("Task not found: %s" % (args['task_name']))
    processor = processors[args['task_name']]()
    args['output_mode'] = output_modes[args['task_name']]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args['label_list'] = label_list
    args['num_labels'] = num_labels

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
    config, tokenizer, model = load_tokenizer_model(args)

    # Training
    if args['do_train']:
        train_dataset = load_and_cache_examples(args, args['task_name'], tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args['do_train'] and (args['local_rank'] == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args['output_dir']) and args['local_rank'] in [-1, 0]:
            os.makedirs(args['output_dir'])

        logger.info("Saving model checkpoint to %s", args['output_dir'])
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args['output_dir'])
        tokenizer.save_pretrained(args['output_dir'])

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args['output_dir'], "training_args.bin"))
        args_json = args
        args_json['device'] = str(args_json['device'])
        json.dump(
            args_json,
            open(os.path.join(args['output_dir'], "training_args.json"), 'w'),
            indent=4
        )

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args['output_dir'])
        tokenizer = tokenizer_class.from_pretrained(args['output_dir'])
        model.to(args['device'])

    # Evaluation
    results = {}
    if args['do_eval'] and args['local_rank'] in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args['output_dir'], do_lower_case=args['do_lower_case'])
        checkpoints = [args['output_dir']]
        if args['eval_all_checkpoints']:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args['device'])
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


# TODO: Clean this code up
def glue_predict(config_path, sent1, sent2, gold_sim=-1):
    '''sent1 and sent2 can be strings (single pred) or lists (multiple preds)'''
    import pandas as pd
    
    args = json.load(open(config_path ,'r'))
    TASK_NAME = args['task_name']
    os.makedirs(f'/tmp/eval_single/{TASK_NAME}', exist_ok=True)
    args['per_gpu_eval_batch_size'] = 1
    args['task_name'] = args['task_name'].lower()
    args['data_dir'] = f'/tmp/eval_single/{TASK_NAME}'
    args['output_mode'] = 'regression'

    # write a sample validation file
    if isinstance(sent1, str) and isinstance(sent2, str): # single prediction
        df = pd.DataFrame(columns=['sent1', 'sent2', 'gold_sim'])
        df = df.append({
            "sent1": sent1,
            "sent2": sent2,
            "gold_sim": gold_sim,
        }, ignore_index=True)
        df.to_csv(f'/tmp/eval_single/{TASK_NAME}/val.tsv', sep='\t', header=True, index=False)
    else: # multiple predictions
        if gold_sim == -1: gold_sim = [-1] * len(sent1)
        df = pd.DataFrame(columns=['sent1', 'sent2', 'gold_sim'])
        for s1, s2, g in zip(sent1, sent2, gold_sim):
            df = df.append({
                "sent1": s1,
                "sent2": s2,
                "gold_sim": g,
            }, ignore_index=True)
        df.to_csv(f'/tmp/eval_single/{TASK_NAME}/val.tsv', sep='\t', header=True, index=False)
        

    # Setup CUDA, GPU & distributed training
    if args['local_rank'] == -1 or args['no_cuda']:
        device = torch.device("cuda" if torch.cuda.is_available() and not args['no_cuda'] else "cpu")
        args['n_gpu'] = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args['local_rank'])
        device = torch.device("cuda", args['local_rank'])
        torch.distributed.init_process_group(backend="nccl")
        args['n_gpu'] = 1
    args['device'] = device
    args['n_gpu'] = torch.cuda.device_count()

    # Load a trained model and vocabulary that you have fine-tuned
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

    model = model_class.from_pretrained(args['model_name_or_path'])
    tokenizer = tokenizer_class.from_pretrained(args['model_name_or_path'])
    model.to(args['device'])

    # Predict
    preds = predict(args, model, tokenizer)
    return preds
