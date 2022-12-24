# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import random
import sys
import time
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import (AutoModel, AutoTokenizer,
                                    LinearDecayWithWarmup)
from tqdm import tqdm, trange

from data_util import convert_example, create_dataloader, read_text_pair
from model import QuestionMatching
from adv import FGM

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler("info.log")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

try:
    paddle.set_device("gpu")
except:
    paddle.set_device("cpu")
    print("No CUDA, use cpu instead.")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path",
                    type=str,
                    required=True,
                    help="The path of model.")
parser.add_argument("--train_set",
                    type=str,
                    required=True,
                    help="The path of train data.")
parser.add_argument("--dev_set",
                    type=str,
                    required=True,
                    help="The path of dev data.")
parser.add_argument("--output_dir",
                    default="./model/checkpoint",
                    type=str,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length",
                    default=256,
                    type=int,
                    help="The maximum total input sequence length after tokenization. "
                         "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--max_steps",
                    default=-1,
                    type=int,
                    help="If > 0,set total number of training steps to perform.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size",
                    default=128,
                    type=int,
                    help="Batch size per GPU/CPU for evaluating.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for optimizer.")
parser.add_argument("--weight_decay",
                    default=0.0,
                    type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs",
                    default=3,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--eval_step",
                    default=100,
                    type=int,
                    help="Step interval for evaluation.")
parser.add_argument("--save_step",
                    default=10000,
                    type=int,
                    help="Step interval for saving checkpoint.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt",
                    type=str,
                    default=None,
                    help="The path of checkpoint to be loaded.")
parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="Random seed for initialization.")
parser.add_argument("--device",
                    choices=["cpu", "gpu"],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument("--rdrop_coef",
                    default=0.0,
                    type=float,
                    help="The coefficient of KL-Divergence loss in R-Drop paper, "
                         "for more detail please refer to https://arxiv.org/abs/2106.14448),"
                         "if rdrop_coef > 0 then R-Drop works")

args = parser.parse_args()


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model,
             metric,
             dataloader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        dataloader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    total_num = 0

    for batch in tqdm(dataloader, desc="Batch"):
        input_ids, token_type_ids, labels = batch
        total_num += len(labels)
        logits, _ = model(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          do_evaluate=True)
        loss = nn.CrossEntropyLoss()(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()

    logger.info("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(
        np.mean(losses), accu, total_num))
    model.train()
    metric.reset()
    return accu


def do_train():
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds = load_dataset(read_text_pair,
                            data_path=args.train_set,
                            is_test=False,
                            lazy=False)

    dev_ds = load_dataset(read_text_pair,
                          data_path=args.dev_set,
                          is_test=False,
                          lazy=False)

    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    train_dataloader = create_dataloader(train_ds,
                                         mode="train",
                                         batch_size=args.train_batch_size,
                                         batchify_fn=batchify_fn,
                                         trans_fn=trans_func)

    dev_dataloader = create_dataloader(dev_ds,
                                       mode="dev",
                                       batch_size=args.eval_batch_size,
                                       batchify_fn=batchify_fn,
                                       trans_fn=trans_func)

    model = QuestionMatching(pretrained_model, rdrop_coef=args.rdrop_coef)
    fgm = FGM(model=model, eps=1)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    model = paddle.DataParallel(model)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_training_steps = len(train_dataloader) * args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    clip = nn.ClipGradByNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=clip
    )

    metric = paddle.metric.Accuracy()

    global_step = 0
    best_score = 0.0

    for _epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        for _step, _batch in enumerate(tqdm(train_dataloader, desc="Iteration"), start=1):
            input_ids, token_type_ids, labels = _batch
            logits1, kl_loss = model(input_ids=input_ids,
                                     token_type_ids=token_type_ids)
            correct = metric.compute(logits1, labels)
            metric.update(correct)
            acc = metric.accumulate()

            ce_loss = nn.CrossEntropyLoss()(logits1, labels)
            if kl_loss > 0:
                loss = ce_loss + kl_loss * args.rdrop_coef
            else:
                loss = ce_loss
            global_step += 1

            loss.backward()

            # FGM
            fgm.attack()
            logits1, kl_loss = model(input_ids=input_ids, token_type_ids=token_type_ids)
            ce_loss = nn.CrossEntropyLoss()(logits1, labels)
            if kl_loss > 0:
                loss_adv = ce_loss + kl_loss * args.rdrop_coef
            else:
                loss_adv = ce_loss

            loss_adv.backward()
            fgm.restore()

            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.eval_step == 0 and rank == 0:
                logger.info("***** Running evaluation *****")
                logger.info("  Epoch = {0} iter {1} step".format(_epoch, global_step))
                logger.info("  Num examples = %d", len(dev_dataloader.dataset))
                logger.info("  Batch size = %d", args.eval_batch_size)
                logger.info("  Learning rate = %.4e", lr_scheduler.get_lr())

                score = evaluate(model, metric, dev_dataloader)

                if score > best_score:
                    best_score = score
                    save_model = True
                else:
                    save_model = False

                if save_model:
                    logger.info("  Saving model to %s after evaluation..." % args.output_dir)

                    save_param_path = os.path.join(args.output_dir,
                                                   "model_state.pdparams")
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(args.output_dir)

            if global_step == args.max_steps:
                return


if __name__ == "__main__":
    do_train()
