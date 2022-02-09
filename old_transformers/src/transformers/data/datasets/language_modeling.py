import os
import pickle
import random
import time
import copy
import json
from typing import Dict, List, Optional
import ast
import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from ...modeling_bert import BertForMaskedLM, BertModel
from ...tokenization_bert import BertTokenizer, BertTokenizerFast

from pathlib import Path
import linecache

# from transformers import BertTokenizer, BertForMaskedLM, BertModel, BertTokenizerFast
# from transformers import BertTokenizer,  BertTokenizerFast
logger = logging.get_logger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

class LineByLineWithWeightTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('|||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and ('||||' not in line)
                                                                             and len(line.split('|||')) ==3 )]
        # temp = list(zip(*lines))
        sents, lm_score, discri_score = list(zip(*lines))
        sents = list(sents)
        discri_score = [ast.literal_eval(x) for x in discri_score]
        lm_score = [float(x) for x in lm_score]

        discri_score_full = torch.tensor(discri_score)
        # just to double check
        print(discri_score_full.exp().sum(dim=1))
        max_values = torch.max(discri_score_full, dim=1).values
        print(max_values.shape)
        thresh = torch.tensor([0.8]).log()
        new_discri_score = []
        new_lm_score = []
        new_sents = []
        todel = []
        for idx, ii in enumerate(max_values.view(-1)):
            if ii < thresh:
                # drop this example.
                todel.append(idx)
            else:
                new_sents.append(sents[idx])
                new_discri_score.append(discri_score[idx])
                new_lm_score.append([lm_score[idx]])
        print(len(todel), len(sents), len(todel)/len(sents))

        sents = new_sents
        discri_score = new_discri_score
        lm_score = new_lm_score


        for i, x in enumerate(sents):
            sents[i] = '[BOS] {} [EOS]'.format(x[6:])
        print(sents[:3])
        print(discri_score[:3])

        batch_encoding = tokenizer(sents, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.labels = copy.deepcopy(self.examples)

        print(self.labels[i])
        print(self.examples[i])
        self.lm_score = lm_score
        self.discri_score = discri_score
        print(self.lm_score[i])
        print(self.discri_score[i])

        print(len(self.labels), len(self.examples), len(self.lm_score), len(self.discri_score))

        print('investigate an easy thing about porportion of GPT2 generation: ')
        idx = torch.max(torch.tensor(self.discri_score), dim=1).indices
        print((idx == 0).sum(), (idx == 1).sum(), (idx == 2).sum(), (idx==3).sum())
        print((idx == 0).sum() / len(self.labels), (idx == 1).sum() / len(self.labels),
              (idx == 2).sum() / len(self.labels), (idx == 3).sum() / len(self.labels))


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            self.lm_score[i], self.discri_score[i])

class LineByLineWithWeightTextDataset_Old(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('###') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('###')) ==3 )]
        temp = list(zip(*lines))
        print(len(temp))
        print(temp[0][:5], temp[1][:5])
        weight, score, sents = list(zip(*lines))
        sents = list(sents)
        batch_encoding = tokenizer(sents, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        separator = tokenizer('[BOS]', add_special_tokens=False)['input_ids'][0]
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx

        print(self.labels[i])
        print(self.examples[i])
        self.weight = [float(w) for w in weight]
        print(self.weight[i])


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            self.weight[i])


class LineByLineText2DataTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, num_layer:int=1):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        edited_sents = []
        for src, tgt in zip(src_lines, tgt_lines):
            sent = ' {} [BOS] '.format(tgt) + src + ' [EOS]'
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]


        separator = tokenizer('[BOS]', add_special_tokens=False)['input_ids'][0]
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),)


# URGENT.
class LineByLineData2TextTextDataset_Sum(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        edited_sents = []
        for src, tgt in zip(src_lines, tgt_lines):
            sent = ' {} {} '.format(src, 'summarize :') + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = []
        for ss in src_lines:
            ssl = [la.split(':')[0].strip() for la in ss.split('|')]
            # print(ssl)
            ssl_lst.append(ssl)

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []
        if True:
            separator = tokenizer(' summarize', add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1])
                self.tgt_sent.append(self.examples[i][sep_idx-1:])
                self.labels[i][:sep_idx] = [-100] * sep_idx


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )

class LineByLineData2TextTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str, lowdata_token:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        if lowdata_token is None:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)
        else:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} {} '.format(lowdata_token, src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = []
        for ss in src_lines:
            ssl = [la.split(':')[0].strip() for la in ss.split('|')]
            # print(ssl)
            ssl_lst.append(ssl)

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1])
                self.tgt_sent.append(self.examples[i][sep_idx-1:])
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx-1
                temp_tgt_len += len(elem) - (sep_idx-1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)




        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )

class LineByLineWritingPromptsTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('|||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('|||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        for i, x in enumerate(src_lines):
            src_lines[i] = x.replace("-lrb-", "(")
            src_lines[i] = x.replace("-rrb-", ")")


        tgt_lines = [x.replace("<newline>", "\n") for x in tgt_lines]

        edited_sents = []
        for src, tgt in zip(src_lines, tgt_lines):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)


        # batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
        #                            is_split_into_words=False)
        if True:
            batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=False,
                                       is_split_into_words=False)
        else:
            batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                                                  is_split_into_words=False)

        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)



        self.src_sent = []
        self.tgt_sent = []
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1])
                self.tgt_sent.append(self.examples[i][sep_idx-1:])
                self.labels[i][:sep_idx] = [-100] * sep_idx


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        # assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                )




class LineByLineSumTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str,
                 max_source_length:int, max_target_length:int, ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        self.src_file = file_path
        self.tgt_file = file_path[:-6] + 'target'
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        eos_idx = tokenizer(eos_tok, add_special_tokens=False)['input_ids'][0]

        self.bos_idx = separator
        self.eos_idx = eos_idx

        self.length = [len(x) for x in Path(self.tgt_file).open().readlines()]
        self.tokenizer = tokenizer
        return



        src_lines = []
        with open(self.src_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0 and not line.isspace():
                    src_lines.append(line)

            # print(len(list(f.read().splitlines())))
            # src_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        print(len(src_lines))
        with open(self.tgt_file, encoding="utf-8") as f:
            tgt_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        print(self.tgt_file, len(tgt_lines), '\n', self.src_file, len(src_lines))

        assert len(tgt_lines) == len(src_lines)

        src_encoding = tokenizer(src_lines, add_special_tokens=True, truncation=True, max_length=max_source_length,
                                                              is_split_into_words=False)['input_ids']

        tgt_encoding = tokenizer(tgt_lines, add_special_tokens=True, truncation=True, max_length=max_target_length,
                                 is_split_into_words=False)['input_ids']

        assert len(src_encoding) == len(tgt_encoding)
        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        eos_idx = tokenizer(eos_tok, add_special_tokens=False)['input_ids'][0]

        edited_sents = []
        for src, tgt in zip(src_encoding, tgt_encoding):
            sent = src + [separator] + tgt + [eos_idx]
            # sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        # batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
        #                                                       is_split_into_words=False)

        self.examples = edited_sents

        self.labels = copy.deepcopy(self.examples)



        self.src_sent = []
        self.tgt_sent = []
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1])
                self.tgt_sent.append(self.examples[i][sep_idx-1:])
                self.labels[i][:sep_idx] = [-100] * sep_idx


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        # assert len(self.src_cat) == len(self.examples)




    def __len__(self):
        return len(self.length)
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        # return (torch.tensor(self.examples[i], dtype=torch.long),
        #         torch.tensor(self.labels[i], dtype=torch.long),
        #         torch.tensor(self.src_sent[i], dtype=torch.long),
        #         torch.tensor(self.tgt_sent[i], dtype=torch.long),
        #         )

        index = i + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        src = self.tokenizer(source_line, add_special_tokens=True, truncation=True, max_length=self.max_source_length,
                                 is_split_into_words=False)['input_ids']

        tgt = self.tokenizer(tgt_line, add_special_tokens=True, truncation=True, max_length=self.max_target_length,
                                 is_split_into_words=False)['input_ids']

        # print(src, tgt)
        sent = src + [self.bos_idx] + tgt + [self.eos_idx]
        # print(sent)

        sep_idx = sent.index(self.bos_idx) + 1

        label = copy.deepcopy(sent)
        label[:sep_idx] = [-100] * sep_idx
        src_sent = sent[:sep_idx - 1]
        tgt_sent = sent[sep_idx - 1:]

        # print(sent)
        # print(label)
        # print(src_sent)
        # print(tgt_sent)
        # print()
        return (torch.tensor(sent, dtype=torch.long),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(src_sent, dtype=torch.long),
                torch.tensor(tgt_sent, dtype=torch.long),
                )


class LineByLineSumBatchGenTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str,
                 max_source_length:int, max_target_length:int, ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        self.src_file = file_path
        self.tgt_file = file_path[:-6] + 'target'
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        eos_idx = tokenizer(eos_tok, add_special_tokens=False)['input_ids'][0]

        self.bos_idx = separator
        self.eos_idx = eos_idx

        self.length = [len(x) for x in Path(self.tgt_file).open().readlines()]
        self.tokenizer = tokenizer
        return




    def __len__(self):
        return len(self.length)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        # return (torch.tensor(self.examples[i], dtype=torch.long),
        #         torch.tensor(self.labels[i], dtype=torch.long),
        #         torch.tensor(self.src_sent[i], dtype=torch.long),
        #         torch.tensor(self.tgt_sent[i], dtype=torch.long),
        #         )

        modegen = 1
        index = i + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        if modegen == 0:

            src = self.tokenizer(source_line, add_special_tokens=True, truncation=True, max_length=self.max_source_length,
                                     is_split_into_words=False)['input_ids']

            tgt = self.tokenizer(tgt_line, add_special_tokens=True, truncation=True, max_length=self.max_target_length,
                                     is_split_into_words=False)['input_ids']

            # print(src, tgt)
            sent = src + [self.bos_idx] + tgt + [self.eos_idx]
            # print(sent)

            sep_idx = sent.index(self.bos_idx) + 1

            label = copy.deepcopy(sent)
            label[:sep_idx] = [-100] * sep_idx
            src_sent = sent[:sep_idx - 1]
            tgt_sent = sent[sep_idx - 1:]

            # print(sent)
            # print(label)
            # print(src_sent)
            # print(tgt_sent)
            # print()
            return (torch.tensor(sent, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long),
                    torch.tensor(src_sent, dtype=torch.long),
                    torch.tensor(tgt_sent, dtype=torch.long),
                    )

        else:
            return (source_line, tgt_line)



class LineByLineWebNLGTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)


        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines_dict['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in sents:
                if sent["comment"] == 'good':
                    full_tgt_lst.append(sent["lex"])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)



        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)


        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0

        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1]) # does not contain the BOS separator
                self.tgt_sent.append(self.examples[i][sep_idx-1:]) # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)




        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        print()
        print(self.labels[1])
        print(self.examples[1])
        print(edited_sents[1])
        print(self.src_sent[1])
        print(self.tgt_sent[1])
        print(self.src_cat[1])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )


class LineByLineTriplesTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)


        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []
        for example in lines_dict:
            rela_lst = []
            temp_triples = ''
            for i, tripleset in enumerate(example['tripleset']):
                subj, rela, obj = tripleset
                rela = rela.lower()
                rela_lst.append(rela)
                if i > 0:
                    temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in example['annotations']:
                full_tgt_lst.append(sent['text'])
                full_src_lst.append(temp_triples)
                full_rela_lst.append(rela_lst)


        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)


        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1]) # does not contain the BOS separator
                self.tgt_sent.append(self.examples[i][sep_idx-1:]) # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx

                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )

class LineByLineLemma2TextTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==3 )]
        src_lines, mid_word, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)
        mid_word = list(mid_word)

        edited_sents = []
        for src, mid, tgt in zip(src_lines, mid_word, tgt_lines):
            sent = ' {} || {} {} '.format(src, mid, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.labels[i][:sep_idx] = [-100] * sep_idx


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),)


class LineByLineClassificationSentimentTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, num_layer:int=1,
                 prefix_ctrl:bool=True, bos_tok='[BOS]', eos_tok='[EOS]'):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('|||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('|||')) ==2 )]
        word_lst, sents = list(zip(*lines))
        sents = list(sents)
        word_lst = list(word_lst)

        sents = [x.replace("< br / >", "\n") for x in sents]

        edited_sents = []
        new_wordlst = []
        for sent, word_temp in zip(sents, word_lst):
            sent = ' {} {} '.format(sent, bos_tok) + word_temp + '{}'.format(eos_tok) # could swap and change to the sentiment is ...
            edited_sents.append(sent)
            new_wordlst.append([word_temp.strip()])

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        # self.prefix_ctrl = prefix_ctrl
        # if prefix_ctrl:
        #     self.control_code = tokenizer(new_wordlst, truncation=True, max_length=block_size,
        #                                   is_split_into_words=True)["input_ids"]


        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])

        # if prefix_ctrl:
        #     print(new_wordlst[0])
        #     # print(tokenizer.decode(self.control_code[0]))
        #     # print(tokenizer.decode([220]))
        #     print(self.control_code[0])

    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long))


class LineByLineSentimentTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, num_layer:int=1,
                 prefix_ctrl:bool=True, bos_tok='[BOS]', eos_tok='[EOS]'):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('|||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('|||')) ==2 )]
        word_lst, sents = list(zip(*lines))
        sents = list(sents)
        word_lst = list(word_lst)

        sents = [x.replace("< br / >", "\n") for x in sents]

        edited_sents = []
        new_wordlst = []
        for sent, word_temp in zip(sents, word_lst):
            sent = ' {}{}'.format(word_temp, bos_tok) + sent + '{}'.format(eos_tok)
            edited_sents.append(sent)
            new_wordlst.append([word_temp.strip()])

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.prefix_ctrl = prefix_ctrl
        if prefix_ctrl:
            self.control_code = tokenizer(new_wordlst, truncation=True, max_length=block_size,
                                          is_split_into_words=True)["input_ids"]


        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])

        if prefix_ctrl:
            print(new_wordlst[0])
            # print(tokenizer.decode(self.control_code[0]))
            # print(tokenizer.decode([220]))
            print(self.control_code[0])

    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.control_code[i], dtype=torch.long) )

class LineByLineClassificationTopicTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, num_layer:int=1,
                 prefix_ctrl:bool=True, bos_tok='[BOS]', eos_tok='[EOS]'):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        word_lst, sents = list(zip(*lines))
        sents = list(sents)
        word_lst = list(word_lst)

        edited_sents = []
        new_wordlst = []
        for sent, word_temp in zip(sents, word_lst):
            sent = ' {} {} '.format(sent, bos_tok) + word_temp + '{}'.format(eos_tok)
            edited_sents.append(sent)
            new_wordlst.append([word_temp.strip()])

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]




        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])



    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long))

class LineByLineTopicTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, num_layer:int=1,
                 prefix_ctrl:bool=True, bos_tok='[BOS]', eos_tok='[EOS]'):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        word_lst, sents = list(zip(*lines))
        sents = list(sents)
        word_lst = list(word_lst)

        edited_sents = []
        new_wordlst = []
        for sent, word_temp in zip(sents, word_lst):
            sent = ' {}{}'.format(word_temp, bos_tok) + sent + '{}'.format(eos_tok)
            # sent = 'Topic {}{}'.format(word_temp, bos_tok) + sent + '{}'.format(eos_tok)
            edited_sents.append(sent)
            new_wordlst.append([word_temp.strip()])

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.prefix_ctrl = prefix_ctrl
        if prefix_ctrl:
            self.control_code = tokenizer(new_wordlst, truncation=True, max_length=block_size,
                                          is_split_into_words=True)["input_ids"]


        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])

        if prefix_ctrl:
            print(new_wordlst[0])
            # print(tokenizer.decode(self.control_code[0]))
            # print(tokenizer.decode([220]))
            print(self.control_code[0])

    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.control_code[i], dtype=torch.long) )


class LineByLineLengthTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, num_layer: int = 1,
                 prefix_ctrl:bool=True):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                            and len(line.split('||')) == 2)]
        word_lst, sents = list(zip(*lines))
        sents = list(sents)
        length_lst = [len(ii.split()) for ii in sents]

        use_dict = True
        edited_sents = []
        word_lst = []
        for sent, len_temp in zip(sents, length_lst):
            if use_dict:
                if len_temp < 10:
                    len_num = 0
                elif len_temp < 20:
                    len_num = 1
                elif len_temp < 30:
                    len_num = 2
                elif len_temp < 40:
                    len_num = 3
                else:
                    len_num = 4
                sent = 'length {} [BOS] '.format(len_num) + sent + ' [EOS]'
                word_lst.append([str(len_num)])
            else:
                sent = 'length {} [BOS] '.format(len_temp) + sent + ' [EOS]'
                word_lst.append([str(len_temp)])
            edited_sents.append(sent)

        self.prefix_ctrl = prefix_ctrl
        if prefix_ctrl:
            self.control_code = tokenizer(word_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                                          is_split_into_words=True)["input_ids"]

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)


        self.examples = batch_encoding["input_ids"]

        separator = tokenizer('[BOS]', add_special_tokens=False)['input_ids'][0]
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx

        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        if prefix_ctrl:
            print(self.control_code[0])

    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.control_code[i], dtype=torch.long) if self.prefix_ctrl else None,)


class LineByLineKeywordTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    # <|endoftext|>

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, num_layer:int=1,
                 prefix_ctrl:bool=True, bos_tok:str='[BOS]', eos_tok:str='[EOS]'):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        word_lst, sents = list(zip(*lines))
        sents = list(sents)
        word_lst = list(word_lst)

        new_wordlst = []
        edited_sents = []
        for sent, word_temp in zip(sents, word_lst):
            sent = 'Keyword {}{}'.format(word_temp, bos_tok) + sent + '{}'.format(eos_tok)
            edited_sents.append(sent)
            new_wordlst.append([word_temp])

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]
        self.prefix_ctrl = prefix_ctrl
        if prefix_ctrl:
            self.control_code = tokenizer(new_wordlst, add_special_tokens=True, truncation=True, max_length=block_size,
                                       is_split_into_words=True)["input_ids"]

        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        print(separator)
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx

        print(self.labels[0])
        print(self.examples[0])
        if prefix_ctrl:
            print(self.control_code[0])
        print(edited_sents[0])


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            torch.tensor(self.control_code[i], dtype=torch.long) if self.prefix_ctrl else None,)

class LineByLineEmbMatchTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, num_layer:int=1,
                 add_bracket:bool = False, bos_tok:str='[BOS]', eos_tok:str='[EOS]'):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)
        self.add_bracket = add_bracket
        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        word_lst, sents = list(zip(*lines))
        sents = list(sents)
        word_lst = list(word_lst)

        full_score, edited_sents = self.get_emb(sents, word_lst, num_layer, bos_tok, eos_tok)
        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=True)
        self.examples = batch_encoding["input_ids"]

        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx

        print(self.labels[i])
        print(self.examples[i])
        print(edited_sents[-1])
        print()
        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        self.emb = torch.split(full_score, 1) # (1, num_layer, 1024) or (1, 1024)
        print(self.emb[i].shape)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            self.emb[i])

    def get_emb(self, sent_lst, word_lst, num_layer, bos_tok, eos_tok):
        # load bert
        tokenizer_bert = BertTokenizerFast.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained('bert-large-uncased', return_dict=True).cuda()
        for param in model.parameters():
            param.requires_grad = False

        device = model.device

        edited_sent = []
        with torch.no_grad():
            computed_ = 0
            mid_ = 300
            full_score = []
            while computed_ < len(sent_lst):
                temp_sent = sent_lst[computed_:computed_+mid_]
                temp_word = word_lst[computed_:computed_+mid_]
                temp_input = tokenizer_bert(temp_sent, return_tensors="pt", padding=True,
                                       is_split_into_words=False, return_offsets_mapping=True, add_special_tokens=True)
                input_ids = temp_input["input_ids"]
                mask_input = temp_input['attention_mask']
                bsz, seqlen = input_ids.shape

                # print(input_ids.shape)


                cand_idx = tokenizer_bert(temp_word, add_special_tokens=False)['input_ids']
                # print(cand_idx)
                # if BPE has multiple subwords.
                cand_idx = torch.tensor([i[-1] for i in cand_idx]) #bsz
                # print(cand_idx)
                cand_idx2 = cand_idx.unsqueeze(1).expand(bsz, seqlen)

                mask = (input_ids == cand_idx2)
                # print(mask.sum(dim=1))
                # print(mask.nonzero())

                # what if the occurence of a subword is not in the primary word?

                # if has multiple occurence? only taking the first one.
                mask = (mask.cumsum(dim=1) == 1) & mask
                # print(mask)
                # print(mask.sum(dim=1))
                # print(mask.nonzero())
                mask_idx = mask.nonzero()


                # print(input_ids.shape)

                edit_temp = []
                keep_mask = []
                for i, (sent1, word1)  in enumerate(zip(temp_sent, temp_word)):
                    # TODO: could check against the offests and make final changes!
                    temp_idx1 = temp_input["offset_mapping"][i][mask_idx[i,1]]
                    # print(word1, sent1)
                    # print(sent1[temp_idx1[0]:temp_idx1[1]])
                    sent1 = sent1.split()
                    widx = sent1.index(word1)
                    by_tokenl = sum([len(l) + 1 for l in sent1[:widx]])
                    by_tokenr = sum([len(l) + 1 for l in sent1[:widx+1]]) - 1
                    # print(by_tokenl, by_tokenr, temp_idx1)
                    if by_tokenl != temp_idx1[0].item() and by_tokenr != temp_idx1[1].item():
                        # print('dangerous')
                        # print(sent1, word1, by_tokenl, by_tokenr, temp_idx1)
                        # simple option: delete it form input_ids
                        keep_mask.append(False)
                        continue
                    else:
                        keep_mask.append(True)

                    if self.add_bracket:
                        new_sent = [word1, bos_tok] + sent1[:widx] + ['[' , sent1[widx], ']'] + sent1[widx+1:] + [eos_tok]
                        assert len(new_sent) == len(sent1) + 5
                    else:
                        new_sent = [word1, bos_tok] + sent1[:widx] + [sent1[widx]] + sent1[widx + 1:] + [eos_tok]
                        assert len(new_sent) == len(sent1) + 3

                    edit_temp.append(new_sent)

                keep_mask = torch.tensor(keep_mask)
                # print(keep_mask.shape, input_ids.shape, mask.shape, 'hi')
                input_ids = input_ids[keep_mask]
                mask = mask[keep_mask]
                mask_input = mask_input[keep_mask]

                # print(input_ids.shape, mask.shape, len(edit_temp))
                assert input_ids.size(0) == len(edit_temp)
                assert len(edit_temp) == mask_input.size(0)

                edited_sent += edit_temp
                print(len(edited_sent), len(mask_input))

                outputs = model(input_ids.to(device), attention_mask=mask_input.to(device), output_hidden_states=True)
                # outputs = model(input_ids.to(device), output_hidden_states=True)

                if num_layer > 1:
                    all_hidden_states = outputs.hidden_states
                    selected_all_hidden_states = [ii[mask] for ii in all_hidden_states[-num_layer:]]
                    # print([ii.shape for ii in selected_all_hidden_states])
                    hidden_layer = torch.stack(selected_all_hidden_states, dim=1)
                    # print(hidden_layer.shape, selected_all_hidden_states[0].shape)
                    # print('all hidden', selected_all_hidden_states.shape)

                else:
                    last_hidden_states = outputs.last_hidden_state
                    hidden_layer = last_hidden_states[mask].unsqueeze(1)


                computed_ += mid_
                full_score.append(hidden_layer.cpu())

            full_score = torch.cat(full_score, dim=0)

        return full_score, edited_sent

class LineByLineWithWeightTextDataset2(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('###') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        weight, sents = list(zip(*lines))
        sents = list(sents)
        for ii, x in enumerate(sents):
            separator = x.find(':')
            sents[ii] = x[:separator] + ' [BOS] ' + x[separator+1:] + ' [EOS]'
        batch_encoding = tokenizer(sents, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        separator = tokenizer('[BOS]', add_special_tokens=False)['input_ids'][0]
        self.labels = copy.deepcopy(self.examples)
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.labels[i][:sep_idx] = [-100] * sep_idx

        print(self.labels[i])
        print(self.examples[i])
        self.weight = [float(w) for w in weight]
        print( self.weight[i])


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i]),
            self.weight[i])

class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int):
        assert os.path.isdir(file_dir)
        logger.info(f"Creating features from dataset file folder at {file_dir}")
        self.examples = []
        # TODO: randomness could apply a random seed, ex. rng = random.Random(random_seed)
        # file path looks like ./dataset/wiki_1, ./dataset/wiki_2
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            assert os.path.isfile(file_path)
            article_open = False
            with open(file_path, encoding="utf-8") as f:
                original_lines = f.readlines()
                article_lines = []
                for line in original_lines:
                    if "<doc id=" in line:
                        article_open = True
                    elif "</doc>" in line:
                        article_open = False
                        document = [
                            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
                            for line in article_lines[1:]
                            if (len(line) > 0 and not line.isspace())
                        ]

                        examples = self.create_examples_from_document(document, block_size, tokenizer)
                        self.examples.extend(examples)
                        article_lines = []
                    else:
                        if article_open:
                            article_lines.append(line)

        logger.info("Dataset parse finished.")

    def create_examples_from_document(self, document, block_size, tokenizer, short_seq_prob=0.1):
        """Creates examples for a single document."""

        # Account for special tokens
        max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        examples = []
        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]  # get a segment
            if not segment:
                i += 1
                continue
            current_chunk.append(segment)  # add a segment to current chunk
            current_length += len(segment)  # overall token length
            # if current length goes to the target length or reaches the end of file, start building token a and b
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A` (first) sentence.
                    a_end = 1
                    # if current chunk has more than 2 sentences, pick part of it `A` (first) sentence
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    # token a
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    # token b
                    tokens_b = []
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if len(tokens_a) == 0 or len(tokens_b) == 0:
                        continue

                    # switch tokens_a and tokens_b randomly
                    if random.random() < 0.5:
                        is_next = False
                        tokens_a, tokens_b = tokens_b, tokens_a
                    else:
                        is_next = True

                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break
                            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                            assert len(trunc_tokens) >= 1
                            # We want to sometimes truncate from the front and sometimes from the
                            # back to add more randomness and avoid biases.
                            if random.random() < 0.5:
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # add special tokens
                    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "sentence_order_label": torch.tensor(0 if is_next else 1, dtype=torch.long),
                    }
                    examples.append(example)
                current_chunk = []  # clear current chunk
                current_length = 0  # reset current text length
            i += 1  # go to next line
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_nsp_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        self.tokenizer = tokenizer

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int):
        """Creates examples for a single document."""

        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if len(current_chunk) == 1 or random.random() < self.nsp_probability:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(0, len(self.documents) - 1)
                            if random_document_index != doc_index:
                                break

                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    self.examples.append(
                        {"tokens_a": tokens_a, "tokens_b": tokens_b, "is_random_next": is_random_next}
                    )

                current_chunk = []
                current_length = 0

            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
