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
import os
import random
import sys
import time
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from model import QuestionMatching
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from tqdm import tqdm, trange

from data_util import convert_example, create_dataloader, read_text_pair

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
parser.add_argument("--input_file",
                    type=str,
                    required=True,
                    help="The full path of input file")
parser.add_argument("--max_seq_length",
                    default=256,
                    type=int,
                    help="The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--device",
                    choices=["cpu", "gpu"],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


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

    print("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(
        np.mean(losses), accu, total_num))
    metric.reset()
    return accu


if __name__ == "__main__":
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

    dev_ds = load_dataset(read_text_pair,
                          data_path=args.input_file,
                          is_test=False,
                          lazy=False)

    dev_dataloader = create_dataloader(dev_ds,
                                       mode="dev",
                                       batch_size=args.batch_size,
                                       batchify_fn=batchify_fn,
                                       trans_fn=trans_func)

    model = QuestionMatching(pretrained_model)

    if args.model_name_or_path and os.path.isfile(os.path.join(args.model_name_or_path, "model_state.pdparams")):
        state_dict = paddle.load(os.path.join(args.model_name_or_path, "model_state.pdparams"))
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % os.path.join(args.model_name_or_path, "model_state.pdparams"))
    else:
        raise ValueError(
            "Please set --model_name_or_path with correct pretrained model file")

    metric = paddle.metric.Accuracy()

    score = evaluate(model, metric, dev_dataloader)
