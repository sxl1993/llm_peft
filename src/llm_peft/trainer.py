#!/usr/bin/env python3
# coding=utf-8
"""
Copyright (c) 2022 Kunlun.com, Inc. All Rights Reserved

LLM PEFT

Authors: shixingliang
Date: 2023/08/22 15:49:00
"""

import os
from typing import Optional

import torch
from peft.peft_model import PeftModel
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class PrefixTrainer(Trainer):
    def __init__(self, *args, save_changed=False, **kwargs):
        """
        Args:
            *args: 父类构造函数中定义的任意位置参数
            save_changed (bool): 是否保存修改，True表示保存，False表示不保存 (默认: False)
            **kwargs: 父类构造函数中定义的任意关键字参数
    
        Returns:
            None
        """
        self.save_changed = save_changed
        super().__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            elif isinstance(self.model, PeftModel):
                logger.info("Saving PeftModel to {}".format(output_dir))
                self.model.save_pretrained(output_dir)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if self.save_changed:
                print("Saving PrefixEncoder")
                state_dict = self.model.state_dict()
                filtered_state_dict = {}
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        filtered_state_dict[k] = state_dict[k]
                self.model.save_pretrained(output_dir, state_dict=filtered_state_dict)
            else:
                print("Saving the whole model")
                self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
