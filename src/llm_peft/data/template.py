#!/usr/bin/env python3
# coding=utf-8
"""
Copyright (c) 2022 Kunlun.com, Inc. All Rights Reserved

LLM PEFT

Authors: shixingliang
Date: 2023/08/22 15:49:00
"""

from dataclasses import dataclass
from typing import Dict, List, Union

# from transformers import PreTrainedTokenizer


@dataclass
class Template:
    """
    Template class
    """
    prompt: List[Union[str, Dict[str, str]]]


templates: Dict[str, Template] = {}

def register_template(
        name: str,
        prompt: List[Union[str, Dict[str, str]]]
    ):
    """
    注册一个新的模板

    Args:
        name (str): 模板名称
        prompt (List[Union[str, Dict[str, str]]]): 模板提示信息，可以是字符串或者字典类型

    Returns:
        None
    """
    templates[name] = Template(prompt)


def get_template_and_fix_tokenizer(
        name: str,
        # tokenizer: PreTrainedTokenizer
    ) -> Template:
    """
    获取模板并修复tokenizer

    Args:
        name (str): 模板名称
        tokenizer (PreTrainedTokenizer): 待修复的tokenizer

    Returns:
        Template: 返回修复好的模板

    Raises:
        AssertionError: 如果模板不存在，则抛出异常。
    """
    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)

    return template


register_template(
    name="chatglm2",
    prompt=[
        "[Round {{idx}}]\n\n问：{{query}}\n\n答："
    ],
)

register_template(
    name="chatglm3",
    prompt=[
        {"token": "<|user|>"},
        "\n",
        "{{query}}",
        {"token": "<|assistant|>"},
        "\n" # add an extra newline to avoid error in ChatGLM's process_response method
    ],
)
