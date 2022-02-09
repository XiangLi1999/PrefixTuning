# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Config class. """

import re
from collections import OrderedDict

from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP, BartConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_bert_generation import BertGenerationConfig
from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from .configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from .configuration_dpr import DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPRConfig
from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from .configuration_encoder_decoder import EncoderDecoderConfig
from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from .configuration_fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig
from .configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig
from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from .configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig
from .configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from .configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
from .configuration_marian import MarianConfig
from .configuration_mbart import MBART_PRETRAINED_CONFIG_ARCHIVE_MAP, MBartConfig
from .configuration_mobilebert import MobileBertConfig
from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from .configuration_pegasus import PegasusConfig
from .configuration_rag import RagConfig
from .configuration_reformer import ReformerConfig
from .configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .configuration_utils import PretrainedConfig
from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from .configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig


ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = dict(
    (key, value)
    for pretrained_map in [
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BART_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MBART_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        T5_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DPR_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ]
    for key, value, in pretrained_map.items()
)


CONFIG_MAPPING = OrderedDict(
    [
        ("retribert", RetriBertConfig),
        ("t5", T5Config),
        ("mobilebert", MobileBertConfig),
        ("distilbert", DistilBertConfig),
        ("albert", AlbertConfig),
        ("bert-generation", BertGenerationConfig),
        ("camembert", CamembertConfig),
        ("xlm-roberta", XLMRobertaConfig),
        ("pegasus", PegasusConfig),
        ("marian", MarianConfig),
        ("mbart", MBartConfig),
        ("bart", BartConfig),
        ("reformer", ReformerConfig),
        ("longformer", LongformerConfig),
        ("roberta", RobertaConfig),
        ("flaubert", FlaubertConfig),
        ("fsmt", FSMTConfig),
        ("bert", BertConfig),
        ("openai-gpt", OpenAIGPTConfig),
        ("gpt2", GPT2Config),
        ("transfo-xl", TransfoXLConfig),
        ("xlnet", XLNetConfig),
        ("xlm", XLMConfig),
        ("ctrl", CTRLConfig),
        ("electra", ElectraConfig),
        ("encoder-decoder", EncoderDecoderConfig),
        ("funnel", FunnelConfig),
        ("lxmert", LxmertConfig),
        ("dpr", DPRConfig),
        ("layoutlm", LayoutLMConfig),
        ("rag", RagConfig),
    ]
)

MODEL_NAMES_MAPPING = OrderedDict(
    [
        ("retribert", "RetriBERT"),
        ("t5", "T5"),
        ("mobilebert", "MobileBERT"),
        ("distilbert", "DistilBERT"),
        ("albert", "ALBERT"),
        ("bert-generation", "Bert Generation"),
        ("camembert", "CamemBERT"),
        ("xlm-roberta", "XLM-RoBERTa"),
        ("pegasus", "Pegasus"),
        ("marian", "Marian"),
        ("mbart", "mBART"),
        ("bart", "BART"),
        ("reformer", "Reformer"),
        ("longformer", "Longformer"),
        ("roberta", "RoBERTa"),
        ("flaubert", "FlauBERT"),
        ("fsmt", "FairSeq Machine-Translation"),
        ("bert", "BERT"),
        ("openai-gpt", "OpenAI GPT"),
        ("gpt2", "OpenAI GPT-2"),
        ("transfo-xl", "Transformer-XL"),
        ("xlnet", "XLNet"),
        ("xlm", "XLM"),
        ("ctrl", "CTRL"),
        ("electra", "ELECTRA"),
        ("encoder-decoder", "Encoder decoder"),
        ("funnel", "Funnel Transformer"),
        ("lxmert", "LXMERT"),
        ("layoutlm", "LayoutLM"),
        ("dpr", "DPR"),
        ("rag", "RAG"),
    ]
)


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: config.__name__ for model_type, config in CONFIG_MAPPING.items()}
        else:
            model_type_to_name = {
                model_type: config_to_class[config].__name__
                for model_type, config in CONFIG_MAPPING.items()
                if config in config_to_class
            }
        lines = [
            f"{indent}- **{model_type}** -- :class:`~transformers.{cls_name}` ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type, cls_name in model_type_to_name.items()
        ]
    else:
        config_to_name = {config.__name__: clas.__name__ for config, clas in config_to_class.items()}
        config_to_model_name = {
            config.__name__: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING.items()
        }
        lines = [
            f"{indent}- :class:`~transformers.{config_name}` configuration class: :class:`~transformers.{cls_name}` ({config_to_model_name[config_name]} model)"
            for config_name, cls_name in config_to_name.items()
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the :meth:`~transformers.AutoConfig.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            "Unrecognized model identifier: {}. Should contain one of {}".format(
                model_type, ", ".join(CONFIG_MAPPING.keys())
            )
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the :obj:`model_type` property of the config
        object that is loaded, or when it's missing, by falling back to using pattern matching on
        :obj:`pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (:obj:`str`):
                Can be either:

                    - A string with the `shortcut name` of a pretrained model configuration to load from cache or
                      download, e.g., ``bert-base-uncased``.
                    - A string with the `identifier name` of a pretrained model configuration that was user-uploaded to
                      our S3, e.g., ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing a configuration file saved using the
                      :meth:`~transformers.PretrainedConfig.save_pretrained` method, or the
                      :meth:`~transformers.PretrainedModel.save_pretrained` method, e.g., ``./my_model_directory/``.
                    - A path or url to a saved configuration JSON `file`, e.g.,
                      ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.,
                :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each
                request.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs(additional keyword arguments, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is
                controlled by the ``return_unused_kwargs`` keyword parameter.

        Examples::

            >>> from transformers import AutoConfig

            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')

            >>> # Download configuration from S3 (user-uploaded) and cache.
            >>> config = AutoConfig.from_pretrained('dbmdz/bert-base-german-cased')

            >>> # If configuration file is in a directory (e.g., was saved using `save_pretrained('./test/saved_model/')`).
            >>> config = AutoConfig.from_pretrained('./test/bert_saved_model/')

            >>> # Load a specific configuration file.
            >>> config = AutoConfig.from_pretrained('./test/bert_saved_model/my_configuration.json')

            >>> # Change some config attributes when loading a pretrained config.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            >>> config.output_attentions
            True
            >>> config, unused_kwargs = AutoConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False, return_unused_kwargs=True)
            >>> config.output_attentions
            True
            >>> config.unused_kwargs
            {'foo': False}
        """
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict:
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **kwargs)
        else:
            # Fallback: use pattern matching on the string.
            for pattern, config_class in CONFIG_MAPPING.items():
                if pattern in pretrained_model_name_or_path:
                    return config_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            "Unrecognized model in {}. "
            "Should have a `model_type` key in its config.json, or contain one of the following strings "
            "in its name: {}".format(pretrained_model_name_or_path, ", ".join(CONFIG_MAPPING.keys()))
        )
