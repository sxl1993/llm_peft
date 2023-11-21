#!/usr/bin/env python3
# coding=utf-8
"""
Copyright (c) 2022 Kunlun.com, Inc. All Rights Reserved

LLM PEFT

Authors: shixingliang
Date: 2023/08/22 15:49:00
"""

import jieba
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from transformers import AutoTokenizer, Seq2SeqTrainingArguments


def compute_metrics(eval_preds, tokenizer: AutoTokenizer, data_args: Seq2SeqTrainingArguments):
    """
    计算模型评估指标。
    
    Args:
        eval_preds (list): 经过解码后的预测结果列表
        tokenizer (AutoTokenizer): 使用的 tokenizer
        data_args (Seq2SeqTrainingArguments): 数据集配置参数
    
    Returns:
        dict: 包含各种评估指标的字典，如 Rouge-1、Rouge-2、Rouge-L 和 BLEU-4 的值
    
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        hypothesis = ' '.join(list(jieba.cut(pred)))
        reference = ' '.join(list(jieba.cut(label)))
        if not hypothesis.strip() or not reference.strip():
            continue
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict