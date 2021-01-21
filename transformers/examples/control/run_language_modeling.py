# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import logging
import math
import os, transformers, torch
from dataclasses import dataclass, field
from typing import Optional
from train_control import PrefixTuning, ClassificationHead, PrefixEmbTuning
from transformers.file_utils import cached_path

import glob





path = os.path.abspath(transformers.__file__)
print(path)

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWeightedLanguageModeling, # modified
    DataCollatorForEmbMatchLanguageModeling, #modified
    DataCollatorForTopicLanguageModeling, #modified
    DataCollatorForLengthLanguageModeling, #modified
    DataCollatorForKeywordLanguageModeling, #modified
    DataCollatorForData2TextLanguageModeling, #modified
    DataCollatorForText2DataLanguageModeling, #modified
    DataCollatorForWritingPromptsLanguageModeling, #modified
    DataCollatorForClassificationSentimentLanguageModeling, #modified
    DataCollatorForSumLanguageModeling, #modified
    HfArgumentParser,
    LineByLineTextDataset,
    LineByLineWithWeightTextDataset, # modified
    LineByLineEmbMatchTextDataset, # modified
    LineByLineTopicTextDataset, # modified
    LineByLineKeywordTextDataset, # modified
    LineByLineLengthTextDataset, # modified
    LineByLineData2TextTextDataset, # modified
    LineByLineLemma2TextTextDataset, # modified
    LineByLineText2DataTextDataset, # modified
    LineByLineTriplesTextDataset, # modified
    LineByLineWebNLGTextDataset,# modified
    LineByLineWritingPromptsTextDataset,# modified
    LineByLineSentimentTextDataset,# modified
    LineByLineClassificationSentimentTextDataset,# modified
    LineByLineClassificationTopicTextDataset,
    LineByLineSumTextDataset,# modified
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    Trainer_Prefix,
    TrainingArguments,
    set_seed,
    GPT2LMHeadModel,
    BertTokenizerFast,
    BertModel,
    AutoModelForSequenceClassification,
    GPT2LMHeadModelAdapter,
)

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
    "length": {
        "path": "/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/pplm/length_classifier_head_epoch_10.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very short":0, "short":1, "medium":2, "long":3, "very long":4},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    }
}

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    prefixModel_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The prefix model checkpoint for weights initialization. "
                    "Leave None if you want to train a model from scratch."
        },
    )

    prefix_mode: Optional[str] = field(
        default='activation',
        metadata={
            "help": "activation or embedding"
        },
    )



    preseqlen: Optional[int] = field(
        default=0,
        metadata={
            "help": "preseqlen for how many tokens of prefix should we include."
        },
    )

    optim_prefix: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether we are optimizing the prefix directly, or optimize another amortized function that "
                    "genrate the prefix."
        },
    )



    tuning_mode: Optional[str] = field(
        default='finetune',
        metadata={
            "help": "whether it's doing prefixtune or finetune."
        },
    )

    objective_mode: Optional[int] = field(
        default=2,
        metadata={
            "help": "In prefixtuning setting, the objective function... "
        },
    )

    top_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": "In finetuning setting, if we only tune the top k layers. "
        },
    )

    adapter_design: Optional[int] = field(
        default=2,
        metadata={
            "help": "For Baseline of the adapter module... (1) means using the NLG adapter reference. "
                    "(2) means using a design similar to adapter module"
        },
    )

    adapter_bottleneck: Optional[int] = field(
        default=100,
        metadata={
            "help": "For baseline adapter module: the mid dim of the adapter. "
        },
    )

    parametrize_emb: Optional[str] = field(
        default='MLP',
        metadata={
            "help": "MLP or Emb to parametrize when we optimize for the embeddings."
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "dropout rate for the prefix tuning model. "
        },
    )

    init_random: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to init a random embedding, or use GPT2 embedding for the prefix tuning model. "
        },
    )

    use_dropout: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to use dropout of GPT2 on trainer. "
        },
    )

    mid_dim: Optional[int] = field(
        default=512,
        metadata={
            "help": "the mid dim."
        },
    )

    dataless_sample_size: Optional[int] = field(
        default=8,
        metadata={
            "help": "the size of samples for each class in dataless training."
        },
    )

    gumbel: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the gumbel softmax trick in training."
        },
    )

    replay_buffer: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the replay buffer in training."
        },
    )

    training_obj: Optional[int] = field(
        default=0,
        metadata={
            "help": "use a specified training objective"
        },
    )


    dataless_sample_length: Optional[int] = field(
        default=20,
        metadata={
            "help": "the length of samples for each class in dataless training."
        },
    )

    dataless_control_type: Optional[int] = field(
        default=0,
        metadata={
            "help": "the type of control in dataless training."
        },
    )

    dataless_usebaseline: Optional[str] = field(
        default='yes',
        metadata={
            "help": "use baseline in dataless training."
        },
    )


    dataless_discri_model_path: Optional[str] = field(
        default='textattack/roberta-base-imdb',
        metadata={
            "help": "the path to discri_model and discri_tokenizer"
        },
    )



    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    task_mode: Optional[str] = field(
        default=None, metadata={"help": "The task mode"}
    )

    format_mode: Optional[str] = field(
        default='cat', metadata={"help": "The mode of data2text format (cat, peek, nopeek)"}
    )

    lowdata_token: Optional[str] = field(
        default='summarize', metadata={"help": "The token to be prepended at initialization time. "}
    )

    use_lowdata_token: Optional[str] = field(
        default='yes', metadata={"help": "Whether we should use the lowdata token and pass it to the prefixTuning Model "
                                         "for the initialization trick.  "}
    )

    dataless: Optional[str] = field(
        default='no', metadata={"help": "Whether we are training or loading dataless model."}
    )

    train_embs: Optional[str] = field(
        default='no', metadata={"help": "whether the train word embeddings"}
    )

    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )

    train_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for training data. "}
    )

    val_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for dev data. "}
    )

    # controlprefix: Optional[str] = field(
    #     default="yes", metadata={"help": "The control mode"}
    # )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    training_args: TrainingArguments = None,
    finetune_mode: bool = False,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        print(args.task_mode)
        # return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
        # return LineByLineWithWeightTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
        if args.task_mode == 'embMatch':
            dataset = LineByLineEmbMatchTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size,
                                                num_layer=1, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'topic':
            dataset = LineByLineTopicTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                    block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'length':
            dataset = LineByLineLengthTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'keyword':
            print(file_path)
            dataset = LineByLineKeywordTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'data2text':
            dataset = LineByLineData2TextTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                 eos_tok=tokenizer.eos_token,
                                                 lowdata_token=args.lowdata_token if ('lowdata' in training_args.output_dir and finetune_mode) else None)

        elif args.task_mode == 'triples':
            dataset = LineByLineTriplesTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                     block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'webnlg':
            dataset = LineByLineWebNLGTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                     block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'writingPrompts':
            dataset = LineByLineWritingPromptsTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                  block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                  eos_tok=tokenizer.eos_token)

        elif args.task_mode =='cnndm' or args.task_mode =='xsum':
            max_source_length = args.max_source_length
            max_target_length = args.train_max_target_length if not evaluate else args.val_max_target_length
            dataset = LineByLineSumTextDataset(tokenizer=tokenizer, file_path=file_path,
                                              block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                              eos_tok=tokenizer.eos_token, max_source_length=max_source_length,
                                               max_target_length=max_target_length,)

        elif args.task_mode == 'sentiment':
            dataset = LineByLineSentimentTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                 eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'classify-sentiment':
            dataset = LineByLineClassificationSentimentTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                 eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'classify-topic':
            dataset = LineByLineClassificationTopicTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                                   block_size=args.block_size,
                                                                   bos_tok=tokenizer.bos_token,
                                                                   eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'lemma2text':
            dataset = LineByLineLemma2TextTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'text2data':
            dataset = LineByLineText2DataTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'gen_data':
            dataset =  LineByLineWithWeightTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                   block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)
        else:
            return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)

        # print(len(dataset))
        # n = len(dataset) % training_args.per_device_train_batch_size
        # if n != 0:
        #     dataset.examples = dataset.examples[:-n]
        #     dataset.labels = dataset.labels[:-n]
        #
        #     if hasattr(dataset, 'emb'):
        #         dataset.emb = dataset.emb[:-n]
        # print(len(dataset))
        return dataset
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
            cache_dir=cache_dir,
        )

def get_classifier(
    name: Optional[str], class_label: int, device: str):
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(class_size=params["class_size"], embed_size=params["embed_size"]).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified in the discriminator model parameters")
    classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # TEMP = (model_args.tuning_mode == 'prefixtune')
    # DATALESS = (data_args.dataless == 'yes')

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    config._my_arg_tune_mode = model_args.tuning_mode

    # 0 means the regular token level objective, which is sum / output_len
    # 1 means the sentence level objective, which is sum
    # 2 means our buggy version which is sum/max_batch(input_len +output_len)
    # 3 means our buggy version which is sum/max_batch(output_len)
    # 4 means our buggy version which is sum/(input_len +output_len)
    config._objective_mode = model_args.objective_mode
    config._my_arg_task_mode = data_args.task_mode

    if model_args.tuning_mode in ['finetune', 'adaptertune', 'finetune-top']:
        print('objective is 0 because of finetune')
    elif model_args.tuning_mode == 'prefixtune':
        print('objective is {}'.format(config._objective_mode ))

    if model_args.tuning_mode == 'adaptertune':
        config.adapter_design = model_args.adapter_design
        config.bottleneck =  model_args.adapter_bottleneck

        if model_args.model_name_or_path:
            config.return_dict = True
            model = GPT2LMHeadModelAdapter.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelWithLMHead.from_config(config)

    else:
        if model_args.model_name_or_path:
            print(config.return_dict)
            config.return_dict = True
            model = GPT2LMHeadModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelWithLMHead.from_config(config)

    # HERE
    # model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # ADD SPECIAL TOKENS:
    if (model_args.tuning_mode != 'prefixtune') and ('lowdata' not in training_args.output_dir) and (model_args.tuning_mode != 'adaptertune'):
        print(model_args.tuning_mode)
        print('adapting the size of the model embedding to include [PAD], [BOS], [EOS].')
        print('len(tokenizer) = ', len(tokenizer))
        num_added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token':'[BOS]', 'eos_token':'[EOS]'})
        embedding_layer = model.resize_token_embeddings(len(tokenizer))
        print('len(tokenizer) = ', len(tokenizer))
    elif data_args.dataless == 'yes':
        print(model_args.tuning_mode, 'dataless setting, so no new tokens at all.')
        print('We do not add special tokens to the tokenizer, instead, we just finetune on <|endoftext|>')

        print(tokenizer.eos_token_id)
        print(tokenizer.eos_token)
        print(tokenizer.pad_token_id)
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer(['he', 'hello w '], padding=True)

        # tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.pad_token, tokenizer.pad_token_id)
    else:
        print(model_args.tuning_mode)
        print('adapting the size of the model embedding to include [PAD]')
        print('len(tokenizer) = ', len(tokenizer))
        num_added_tokens = tokenizer.add_special_tokens(
            {'pad_token': '[PAD]'})
        embedding_layer = model.resize_token_embeddings(len(tokenizer))
        print('len(tokenizer) = ', len(tokenizer))
        print(tokenizer.eos_token, tokenizer.eos_token_id)
        print(tokenizer.bos_token, tokenizer.bos_token_id)

    if model_args.tuning_mode == 'prefixtune': # prefixtune
        for param in model.base_model.parameters():
            param.requires_grad = False

        gpt2 = model



        print('loading the prefix model from ', model_args.prefixModel_name_or_path)
        # print(bool(".ckpt" in model_args.prefixModel_name_or_path))
        if model_args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif model_args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if model_args.prefixModel_name_or_path is not None:
            config2 = AutoConfig.from_pretrained(model_args.prefixModel_name_or_path, cache_dir=model_args.cache_dir)
            # print(config2)

            if model_args.prefix_mode == 'embedding':
                model = PrefixEmbTuning.from_pretrained(
                        model_args.prefixModel_name_or_path,
                        from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                        config=config2,
                        cache_dir=model_args.cache_dir,
                        model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                        use_infix=(data_args.format_mode == 'infix')
                    )

            elif model_args.prefix_mode == 'activation':

                model = PrefixTuning.from_pretrained(
                    model_args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                    config=config2,
                    cache_dir=model_args.cache_dir,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                    use_infix=(data_args.format_mode == 'infix')
                )
            else:
                assert False, "invalid prefix mode"

        else:

            # should clone the config and construct it.
            config_prefix = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
            config_prefix._my_arg_tune_mode = model_args.tuning_mode
            config_prefix._my_arg_task_mode = data_args.task_mode
            config_prefix._my_arg_control = True
            config_prefix.train_weights = data_args.train_embs
            config_prefix.optim_prefix = optim_prefix_bool
            config_prefix.preseqlen = model_args.preseqlen
            config_prefix.use_infix = (data_args.format_mode == 'infix')
            config_prefix.format_mode = data_args.format_mode
            config_prefix.prefix_dropout = model_args.prefix_dropout
            config_prefix.vocab_size = len(tokenizer)
            config_prefix.lowdata = ('lowdata' in training_args.output_dir)
            if config_prefix.lowdata and data_args.use_lowdata_token == 'yes':
                config_prefix.lowdata_token = tokenizer([data_args.lowdata_token],
                                                        add_prefix_space=True)['input_ids']  #return_tensors='np',
                print(data_args.lowdata_token)
                print(config_prefix.lowdata_token)

            # some extra stuff.
            config_prefix.init_random = model_args.init_random
            config_prefix.mid_dim = model_args.mid_dim



            print('training the prefix model from scratch. ')
            if model_args.prefix_mode == 'embedding':

                # specific parametrization for embedding.
                config_prefix.parametrize_emb = model_args.parametrize_emb

                model = PrefixEmbTuning(config_prefix, model_gpt2=gpt2)

                # model = PrefixEmbTuning(config, model_gpt2=gpt2,
                #                         optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                #                         use_infix=(data_args.format_mode == 'infix'))
            elif model_args.prefix_mode == 'activation':
                model = PrefixTuning(config_prefix, model_gpt2=gpt2)

                # model = PrefixTuning(config, model_gpt2=gpt2,
                #                      optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                #                      use_infix=(data_args.format_mode == 'infix'))
            else:
                assert False, "invalid prefix mode"

        if (data_args.dataless == 'yes'):
            print('in dataless setting, loading the discriminator. ')
            if model_args.dataless_control_type == 0:
                discri_model, _ = get_classifier('sentiment', 1, training_args.device )
                discri_tokenizer = None
                discri_labels = ['positive', 'negative']
            elif model_args.dataless_control_type == 1:
                discri_tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
                discri_model = BertModel.from_pretrained('bert-large-uncased', return_dict=True).cuda()
                for param in discri_model.parameters():
                    param.requires_grad = False
            elif model_args.dataless_control_type == 2:
                # using pretrained models of BERT/Albert/Robert etc...
                discri_config = AutoConfig.from_pretrained(model_args.dataless_discri_model_path)
                print(discri_config)
                print('loading the discriminator from', model_args.dataless_discri_model_path)
                discri_tokenizer = AutoTokenizer.from_pretrained(model_args.dataless_discri_model_path)
                discri_model = AutoModelForSequenceClassification.from_pretrained(
                    model_args.dataless_discri_model_path, return_dict=True).cuda()
                # print(discri_model.class_labels)
                for param in discri_model.parameters():
                    param.requires_grad = False

                # a quick check
                # 1 means positive and 0 means negative
                if model_args.dataless_discri_model_path == 'textattack/roberta-base-imdb':
                    temp_input = discri_tokenizer('I am happy.', return_tensors="pt", add_special_tokens=True)
                    temp_result = discri_model(temp_input['input_ids'].to(discri_model.device)).logits.view(-1).data
                    print(temp_result)
                    if temp_result[0] < temp_result[1]:
                        print('1 means positive and 0 means negative')
                        discri_labels = ['negative', 'positive']
                    else:
                        print('0 means positive and 1 means negative')
                        discri_labels = None
                    assert temp_result[0] < temp_result[1]
                elif model_args.dataless_discri_model_path == 'textattack/roberta-base-ag-news':
                    discri_labels = ['world', 'sports', 'business', 'science']
                else:
                    discri_labels = [x for x in discri_config.label2id.keys()]
            elif model_args.dataless_control_type == 3:
                print('controlling the length of generation. ')
                discri_labels = ['short', 'medium', 'long']
                discri_model = torch.tensor([5, 25, 45])
                # discri_model = None
                discri_tokenizer = None
            print(model)
        else:
            print('Not in dataless setting, loading the control code. ')
            if 'sentiment' in  training_args.output_dir:
                print('sentiment does need discri_labels')
                discri_labels = None
            elif 'classify-sentiment' in training_args.output_dir:
                print('classify-sentiment does need discri_labels')
                discri_labels = None
            elif 'classify-topic' in training_args.output_dir:
                print('classify-topic does need discri_labels')
                discri_labels = None
            elif 'sent' in training_args.output_dir:
                discri_labels = ['negative', 'positive']
            elif 'topic' in training_args.output_dir:
                discri_labels = ['world', 'sports', 'business', 'science']
            elif 'keyword' in training_args.output_dir:
                print('keyword is unbounded.')
                discri_labels = None
            elif 'embMatch' in training_args.output_dir:
                print('embMatch is unbounded.')
                discri_labels = None
            elif 'data2text' in training_args.output_dir:
                print('data2text does need discri_labels' )
                discri_labels = None
            elif 'triples' in training_args.output_dir:
                print('triples does need discri_labels' )
                discri_labels = None
            elif 'webnlg' in training_args.output_dir:
                print('triples does need discri_labels' )
                discri_labels = None
            elif 'writingPrompts' in training_args.output_dir:
                print('writingPrompts does need discri_labels')
                discri_labels = None
            elif 'cnndm' in training_args.output_dir:
                print('cnndm does need discri_labels')
                discri_labels = None
            elif 'xsum' in training_args.output_dir:
                print('xsum does need discri_labels')
                discri_labels = None
            elif 'lemma2text' in training_args.output_dir:
                print('lemma2text does need discri_labels' )
                discri_labels = None
            else:
                assert False, 'should have topic/sent in the file name'

    elif model_args.tuning_mode == 'finetune-top':
        # print(model.config)
        # print(model)
        for param in model.base_model.parameters():
            param.requires_grad = False

        top_layers = model_args.top_layers
        total_params = 0
        if top_layers == 0:
            for name, param in model.named_parameters():
                if 'transformer.ln_f.' in name or 'transformer.wte' in name:
                    print(name)
                    param.requires_grad = True
                    total_params += param.numel()
        elif top_layers == 1:
            for name, param in model.named_parameters():
                if 'transformer.ln_f.' in name or 'transformer.wte' in name or 'transformer.h.23.' in name:
                    print(name)
                    param.requires_grad = True
                    total_params += param.numel()

        elif top_layers == 2:
            for name, param in model.named_parameters():
                if 'transformer.ln_f.' in name or 'transformer.wte' in name or 'transformer.h.23.' in name or \
                        'transformer.h.22.' in name:
                    print(name)
                    param.requires_grad = True
                    print(param.shape, param.numel())
                    total_params += param.numel()

        elif top_layers == 22:
            for name, param in model.named_parameters():
                if 'transformer.ln_f.' in name or 'transformer.h.23.' in name or \
                        'transformer.h.22.' in name:
                    print(name)
                    param.requires_grad = True
                    print(param.shape, param.numel())
                    total_params += param.numel()

        elif top_layers == 11:
            for name, param in model.named_parameters():
                if 'transformer.ln_f.' in name or 'transformer.h.23.' in name:
                    print(name)
                    param.requires_grad = True
                    print(param.shape, param.numel())
                    total_params += param.numel()

        elif top_layers == 00:
            for name, param in model.named_parameters():
                if 'transformer.ln_f.' in name:
                    print(name)
                    param.requires_grad = True
                    print(param.shape, param.numel())
                    total_params += param.numel()
        print('the total number of trainable parameters is {}'.format(total_params))


    elif model_args.tuning_mode == 'adaptertune':
        print(model_args.tuning_mode)

        for param in model.base_model.parameters():
            param.requires_grad = False

        total_params = 0
        for name, param in model.named_parameters():
            if 'adapter_block' in name:
                print(name, end=' ')
                param.requires_grad = True
                print(param.shape, param.numel())
                total_params += param.numel()

        print('the total number of trainable parameters is {}'.format(total_params))


        # return



    elif model_args.tuning_mode == 'bothtune': # prefixtune
        print('IN BOTH TUNE: DOING both prefixtuning and the finetuning.')
        for param in model.base_model.parameters():
            param.requires_grad = True

        gpt2 = model

        discri_labels=None


        print('loading the prefix model from ', model_args.prefixModel_name_or_path)
        # print(bool(".ckpt" in model_args.prefixModel_name_or_path))
        if model_args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif model_args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if model_args.prefixModel_name_or_path is not None:
            config2 = AutoConfig.from_pretrained(model_args.prefixModel_name_or_path, cache_dir=model_args.cache_dir)
            # print(config2)

            if model_args.prefix_mode == 'embedding':
                model = PrefixEmbTuning.from_pretrained(
                        model_args.prefixModel_name_or_path,
                        from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                        config=config2,
                        cache_dir=model_args.cache_dir,
                        model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                        use_infix=(data_args.format_mode == 'infix')
                    )

            elif model_args.prefix_mode == 'activation':

                model = PrefixTuning.from_pretrained(
                    model_args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                    config=config2,
                    cache_dir=model_args.cache_dir,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                    use_infix=(data_args.format_mode == 'infix')
                )
            else:
                assert False, "invalid prefix mode"

        else:

            # should clone the config and construct it.
            config_prefix = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
            config_prefix._my_arg_tune_mode = model_args.tuning_mode
            config_prefix._my_arg_task_mode = data_args.task_mode
            config_prefix._my_arg_control = True
            config_prefix.train_weights = data_args.train_embs
            config_prefix.optim_prefix = optim_prefix_bool
            config_prefix.preseqlen = model_args.preseqlen
            config_prefix.use_infix = (data_args.format_mode == 'infix')
            config_prefix.format_mode = data_args.format_mode
            config_prefix.prefix_dropout = model_args.prefix_dropout
            config_prefix.vocab_size = len(tokenizer)
            config_prefix.lowdata = ('lowdata' in training_args.output_dir)
            if config_prefix.lowdata and data_args.use_lowdata_token == 'yes':
                config_prefix.lowdata_token = tokenizer([data_args.lowdata_token],
                                                        add_prefix_space=True)['input_ids']  #return_tensors='np',
                print(data_args.lowdata_token)
                print(config_prefix.lowdata_token)

            # some extra stuff.
            config_prefix.init_random = model_args.init_random
            config_prefix.mid_dim = model_args.mid_dim



            print('training the prefix model from scratch. ')
            if model_args.prefix_mode == 'embedding':
                config_prefix.parametrize_emb = model_args.parametrize_emb

                model = PrefixEmbTuning(config_prefix, model_gpt2=gpt2)

            elif model_args.prefix_mode == 'activation':
                model = PrefixTuning(config_prefix, model_gpt2=gpt2)

            else:
                assert False, "invalid prefix mode"











    # Get datasets
    if data_args.task_mode == 'generate':

        prompt_text = '[BOS] By the riverside, '

        if prompt_text == '':
            input_ids_prompt = None
        else:
            input_ids_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(training_args.device)

        print(input_ids_prompt)

        discri_labels_code = tokenizer(discri_labels, return_tensors="pt",
                                       is_split_into_words=True, add_special_tokens=False)['input_ids']
        discri_labels_code = discri_labels_code.view(-1).to(training_args.device).unsqueeze(0).split(1, dim=1)
        print(discri_labels_code)
        model = model.to(training_args.device)
        gpt2 = gpt2.to(training_args.device)

        quick_generate(discri_labels, discri_labels_code, input_ids_prompt, prompt_text, model, gpt2, tokenizer, sample_size=10, sample_from_gpt=False,
                   textlength=50, nolinebreak=True)
        return

    if (data_args.dataless == 'yes'):
        pass
    else:
        train_dataset = (
            get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, training_args=training_args,
                        finetune_mode=(model_args.tuning_mode == 'finetune')) if training_args.do_train else None
        )
        eval_dataset = (
            get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir,
                        training_args=training_args, finetune_mode=(model_args.tuning_mode == 'finetune') )
            if training_args.do_eval
            else None
        )
        if config.model_type == "xlnet":
            data_collator = DataCollatorForPermutationLanguageModeling(
                tokenizer=tokenizer,
                plm_probability=data_args.plm_probability,
                max_span_length=data_args.max_span_length,
            )
        else:

            if data_args.task_mode == 'embMatch':
                data_collator = DataCollatorForEmbMatchLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
                )
            elif data_args.task_mode == 'topic' or data_args.task_mode == 'sentiment':
                data_collator = DataCollatorForKeywordLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
                )
            elif data_args.task_mode == 'classify-topic' or data_args.task_mode == 'classify-sentiment':
                data_collator = DataCollatorForClassificationSentimentLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
                )
            elif data_args.task_mode == 'length':
                data_collator = DataCollatorForKeywordLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
                )
            elif data_args.task_mode == 'keyword':
                data_collator = DataCollatorForKeywordLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
                )
            elif data_args.task_mode == 'data2text' or data_args.task_mode== 'triples' or data_args.task_mode == 'webnlg':
                print('FORMAT MODE IS ', data_args.format_mode)
                data_collator = DataCollatorForData2TextLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
                    format_mode=data_args.format_mode
                )
            elif data_args.task_mode == 'writingPrompts':
                print('FORMAT MODE IS ', data_args.format_mode)
                data_collator = DataCollatorForWritingPromptsLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
                    format_mode=data_args.format_mode
                )
            elif data_args.task_mode == 'xsum' or  data_args.task_mode == 'cnndm':
                print('FORMAT MODE IS ', data_args.format_mode)
                data_collator = DataCollatorForSumLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
                    format_mode=data_args.format_mode
                )
            elif data_args.task_mode == 'lemma2text':
                data_collator = DataCollatorForData2TextLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
                )
            elif data_args.task_mode == 'text2data':
                data_collator = DataCollatorForText2DataLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
                )
            elif data_args.task_mode == 'gen_data':
                data_collator = DataCollatorForWeightedLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
                )
            else:
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
                )
            # data_collator = DataCollatorForWeightedLanguageModeling(
            #     tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            # )



    # Initialize our Trainer
    # HERE!
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     prediction_loss_only=True,
    # )

    if (model_args.tuning_mode == 'prefixtune'):
        if (data_args.dataless == 'yes'):
            trainer = Trainer_Prefix(
                model=model,
                model_gpt2=gpt2,
                args=training_args,
                prediction_loss_only=True,
                tokenizer=tokenizer,
                discri_tokenizer = discri_tokenizer,
                discri_model=discri_model,
                dataless_sample_size=model_args.dataless_sample_size,
                dataless_sample_length=model_args.dataless_sample_length,
                dataless_control_type=model_args.dataless_control_type,
                dataless_usebaseline= (model_args.dataless_usebaseline == 'yes'),
                discri_labels=discri_labels,
                gumbel=(model_args.gumbel == 'yes'),
                replay_buffer = (model_args.replay_buffer == 'yes'),
                forward_kl='no', # focus of this week.
                reverse_kl='yes',
                sample_from_gpt = False, # true for forward_kl is ok. cannot be true for reverse KL.

            )
        else:
            if 'topic' in training_args.output_dir:
                discri_labels = ['world', 'sports', 'business', 'science']
            elif 'sent' in training_args.output_dir:
                discri_labels = ['negative', 'positive']
            trainer = Trainer_Prefix(
                model=model,
                tokenizer=tokenizer,
                discri_labels=discri_labels,
                model_gpt2=gpt2,
                args=training_args,
                prediction_loss_only=True,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                task_mode =data_args.task_mode,
                use_dropout=(model_args.use_dropout == 'yes')
            )

    elif (model_args.tuning_mode == 'bothtune'):
        print('BOTH TUNE for trainer prefix. ')
        trainer = Trainer_Prefix(
            model=model,
            tokenizer=tokenizer,
            discri_labels=discri_labels,
            model_gpt2=gpt2,
            args=training_args,
            prediction_loss_only=True,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            task_mode=data_args.task_mode,
            use_dropout=(model_args.use_dropout == 'yes'),
            both_tune=True,
        )

    else:

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
        )

    # Training

    if 'lowdata' in training_args.output_dir:
        eval_output = trainer.evaluate()
        # perplexity = math.exp(eval_output["eval_loss"])
        print('initial eval loss is {}'.format(eval_output["eval_loss"]))

    if False:
        # data collection:
        print('collecting data to the files {}/gptgen_sentiment.txt'.format(training_args.output_dir))
        out_path = '{}/gptgen_sentiment.txt'.format(training_args.output_dir)
        trainer.gen_data(out_path)
    elif training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        if not (data_args.dataless == 'yes'):
            trainer.train(model_path=model_path)
        elif False:
            trainer.train_dataless(model_path=model_path, verbose=True)
        else:
            trainer.train_amortized_pplm(model_path=model_path, verbose=True)

        if 'lowdata' not in training_args.output_dir:
            trainer.save_model()

            if model_args.tuning_mode == 'bothtune':
                gpt2_dir = os.path.join(training_args.output_dir, 'gpt2')
                gpt2.save_pretrained(gpt2_dir)

        # # For convenience, we also re-save the tokenizer to the same directory,
        # # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
        #     tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and not (data_args.dataless == 'yes'):
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)


    if 'lowdata' in training_args.output_dir:
        print('evaluating the PPL on full dev data. ')
        data_args.eval_data_file = "/u/scr/xlisali/e2e_data/src1_valid.txt"
        eval_dataset = (
            get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir,
                        training_args=training_args, finetune_mode=(model_args.tuning_mode == 'finetune'))
            if training_args.do_eval
            else None
        )
        print(len(eval_dataset))
        eval_output = trainer.evaluate(eval_dataset)
        perplexity = math.exp(eval_output["eval_loss"])
        print('                full_dev_perplexity = {}'.format(perplexity))



        del model
        del trainer
        torch.cuda.empty_cache()
        # gpt2 = gpt2.cpu()
        elem = os.path.abspath(training_args.output_dir)
        checkpoint_path = glob.glob(os.path.join(elem, '*checkpoint*'))
        assert len(checkpoint_path) == 1
        checkpoint_path = checkpoint_path[0]

        print('running evaluation on ', checkpoint_path)

        os.system('python ../text-generation/gen.py data2text yes yes {} no'.format(checkpoint_path))

    elif data_args.task_mode == 'data2text':
        del model
        del trainer
        if model_args.tuning_mode == 'prefixtune' or model_args.tuning_mode == 'bothtune' :
            del gpt2
        torch.cuda.empty_cache()
        elem = os.path.abspath(training_args.output_dir)
        checkpoint_path = elem

        print('running evaluation on ', checkpoint_path)

        os.system('python ../text-generation/gen.py data2text yes yes {} no'.format(checkpoint_path))

        if 'earlystop' in  training_args.output_dir:
            elem = os.path.abspath(training_args.output_dir)
            checkpoint_path = glob.glob(os.path.join(elem, '*checkpoint*'))
            assert len(checkpoint_path) == 1
            checkpoint_path = checkpoint_path[0]

            print('running early stopping evaluation on ', checkpoint_path)

            os.system('python ../text-generation/gen.py data2text yes yes {} no'.format(checkpoint_path))


    elif data_args.task_mode == 'webnlg':
        del model
        del trainer
        if model_args.tuning_mode == 'prefixtune':
            del gpt2
        torch.cuda.empty_cache()
        elem = os.path.abspath(training_args.output_dir)
        checkpoint_path = elem

        print('running evaluation on ', checkpoint_path)

        os.system('python ../text-generation/gen.py webnlg yes yes {} no'.format(checkpoint_path))


        # also run for early stopping:
        if 'earlystop' in  training_args.output_dir:
            elem = os.path.abspath(training_args.output_dir)
            checkpoint_path = glob.glob(os.path.join(elem, '*checkpoint*'))
            assert len(checkpoint_path) == 1
            checkpoint_path = checkpoint_path[0]

            print('running early stopping evaluation on ', checkpoint_path)

            os.system('python ../text-generation/gen.py webnlg yes yes {} no'.format(checkpoint_path))
        
        
    elif data_args.task_mode == 'triples':
        del model
        del trainer
        if model_args.tuning_mode == 'prefixtune':
            del gpt2
        torch.cuda.empty_cache()
        elem = os.path.abspath(training_args.output_dir)
        checkpoint_path = elem

        print('running evaluation on ', checkpoint_path)

        os.system('python ../text-generation/gen.py triples yes yes {} no'.format(checkpoint_path))


        if 'earlystop' in  training_args.output_dir:
            elem = os.path.abspath(training_args.output_dir)
            checkpoint_path = glob.glob(os.path.join(elem, '*checkpoint*'))
            assert len(checkpoint_path) == 1
            checkpoint_path = checkpoint_path[0]

            print('running early stopping evaluation on ', checkpoint_path)

            os.system('python ../text-generation/gen.py triples yes yes {} no'.format(checkpoint_path))


    return results

def quick_generate(discri_labels, discri_labels_code, input_ids_prompt, prompt_text, model, gpt2, tokenizer, sample_size=10, sample_from_gpt=False,
                   textlength=50, nolinebreak=True, stop_token='[EOS]'):
    control_codes = []
    sst_codes = []
    prompt_codes = []
    for a in range(len(discri_labels)):
        sst_label = discri_labels[a]
        control_code = discri_labels_code[a]
        control_codes += [control_code] * sample_size
        sst_codes += [a] * sample_size
        if not sample_from_gpt:
            prompt = model.get_prompt(control_code, gpt2)
            prompt = [x.expand(-1, sample_size, -1, -1, -1) for x in
                      prompt]  # (2, batch_size, num_heads, sequence_length, embed_size_per_head)
        else:
            prompt = None
        # print(len(prompt), prompt[0].shape)
        prompt_codes.append(prompt)

    if not sample_from_gpt:
        prompt_codes = list(zip(*prompt_codes))
        # print(len(prompt_codes), len(prompt_codes[0]), prompt_codes[0][0].shape)
        prompt_full = []
        for prompt_c in prompt_codes:
            # print(len(prompt_c), prompt_c[0].shape, prompt_c[1].shape)
            prompt_c = torch.cat(prompt_c, dim=1)
            prompt_full.append(prompt_c)
    else:
        prompt_full = None

    full_results = gpt2.generate(input_ids=input_ids_prompt,
                                      emb_match=None,
                                      control_code=None,
                                      past_key_values=prompt_full,
                                      max_length=textlength,
                                      temperature=1.0,
                                      top_k=0,
                                      top_p=0.9,
                                      repetition_penalty=1.0,
                                      do_sample=True,
                                      num_return_sequences=sample_size * len(discri_labels),
                                      bad_words_ids=[[628], [198]] if nolinebreak else None,
                                      use_cache=True)

    print(full_results)

    for generated_sequence_idx, generated_sequence in enumerate(full_results):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        if input_ids_prompt is not None:
            total_sequence = (
                    prompt_text + text[len(tokenizer.decode(input_ids_prompt[0], clean_up_tokenization_spaces=True)):]
            )
        else:
            total_sequence = (
                    text
            )

        print(discri_labels[sst_codes[generated_sequence_idx]])
        # generated_sequences.append(total_sequence)
        print(total_sequence)

    print()

    # clean up samples...

    # full_results.append(results)

    # sst_codes = torch.LongTensor(sst_codes)
    # # full_results = self._tensorize_batch(full_results, self.tokenizer.eos_token_id)
    #
    # control_codes = torch.cat(control_codes, dim=0)
    # labels = full_results.clone()
    # if self.tokenizer.eos_token_id is not None:
    #     mask = (labels == self.tokenizer.eos_token_id)
    #     mask_cumsum = mask.cumsum(1)
    #     mask = (mask & (mask_cumsum != 1) & (mask_cumsum != 2))
    #     labels[mask] = -100
    # return {'input_ids': full_results, 'control_code': control_codes, 'labels': labels, 'sst_codes': sst_codes}
    return {'input_ids': full_results, 'control_code': control_codes, 'sst_codes': sst_codes}
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":

    main()
