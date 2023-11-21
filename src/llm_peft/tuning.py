#!/usr/bin/env python3
# coding=utf-8
"""
Copyright (c) 2022 Kunlun.com, Inc. All Rights Reserved

LLM PEFT

Authors: shixingliang
Date: 2023/08/22 15:49:00
"""

import warnings
from typing import Optional

import torch
from peft import (LoraConfig, PeftType, PrefixTuningConfig, TaskType,
                  get_peft_model)
from peft.config import PeftConfig
from peft.peft_model import PeftModel, PeftModelForCausalLM
from peft.utils.other import _get_batch_size


class PeftModelForChatGLM(PeftModelForCausalLM):
    def get_prompt(self, batch_size: int, task_ids: Optional[torch.Tensor] = None):
        """
        返回用于Peft的虚拟提示。仅当`peft_config.peft_type != PeftType.LORA`时才适用。
    
        Args:
            batch_size (int): 批处理大小。
            task_ids (Optional[torch.Tensor]): 任务ID张量（可选）。
    
        Returns:
            prompt_values (torch.Tensor): 用于Peft的虚拟提示张量。
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_torch_dtype)
            past_key_values = past_key_values.view(
                batch_size,
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)

            # 适配chatglm2-6b
            past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(
                peft_config.num_transformer_submodules * 2
            )
            return past_key_values
        else:
            if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
                prompts = prompt_encoder(prompt_tokens, task_ids)
            else:
                if peft_config.inference_mode:
                    prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
                else:
                    prompts = prompt_encoder(prompt_tokens)
            return prompts

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        """
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Indices of input sequence tokens in the vocabulary. Padding will be applied if the expected input
                dimension is not相适应。
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are NOT MASKED
                - 0 for MASKED tokens.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, 
                `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
                (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` object instead of a plain
                Python tuple.
            task_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Task IDs for each token in the sequence. These values allow to retrieve the task label for each token.
                If set to None, the method will not return task labels. Note that the batch size should be set to 1 for
                this method to work properly.
            kwargs (:obj:`Dict[str, any]`, optional):
                Optional keyword arguments passed along to the specific model forward method.
        
        Returns:
            :class:`~transformers.file_utils.ModelOutput` or :obj:`torch.Tensor`: A :class:`~transformers.file_utils.ModelOutput`
            object (if ``return_dict=True``) or a :obj:`torch.Tensor` (if ``return_dict=False`` and returns a loss or a
            logits tensor).
            """
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = _get_batch_size(input_ids, inputs_embeds)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        """
        根据给定的参数准备输入以进行生成。
    
        Args:
            *args: 其他未列出的参数。
            task_ids (:obj:`torch.Tensor`, 可选): 任务的标识符。
            **kwargs: 关键字参数列表。
    
        Returns:
            :obj:`Dict[str, torch.Tensor]`: 包含输入张量的字典。
    
        Raises:
            无。
        """
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.is_prompt_learning:
            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None

        return model_kwargs

def get_prefix_tuning2_model(model, model_name, pre_seq_len, token_dim, num_attention_heads):
    """
    根据给定的模型名称和配置，返回带有前缀调谐（PrefixTuning）的预训练模型。

    Args:
        model (Union[str, Model]): 预训练模型的名称或已加载的预训练模型对象。
        model_name (str): 预训练模型的名称。
        pre_seq_len (int): 序列长度，即需要预测的单词的前面单词序列长度。
        token_dim (int): 每个token的维度。
        num_attention_heads (int): self-attention中并行的头数。

    Returns:
        Union[Model, str]: 带有前缀调谐的预训练模型或已加载的预训练模型对象。
    """
    config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                num_virtual_tokens=pre_seq_len,
                                token_dim=token_dim,
                                num_attention_heads=num_attention_heads,
                                inference_mode=False)
    if model_name not in ["chatglm2-6b", "chatglm3-6b"]:
        model = get_peft_model(model, config)
    return model

def get_lora_model(model, model_name, target_modules, lora_rank, lora_dropout):
    model_name = model_name
    target_modules = target_modules.split(",")
    config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                        lora_alpha=2 * lora_rank,
                        target_modules=target_modules,
                        inference_mode=False,
                        r=lora_rank,
                        lora_dropout=lora_dropout)
    model = get_peft_model(model, config)
    return model
