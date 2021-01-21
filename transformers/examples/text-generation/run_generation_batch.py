
#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging

import numpy as np
import torch
import json
from torch.utils.data import DataLoader
import time
from rouge_score import rouge_scorer, scoring
import itertools
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    BertForMaskedLM, BertModel,
    BertTokenizer, BertTokenizerFast, AutoConfig,
    set_seed,
    GPT2LMHeadModelAdapter,
    LineByLineSumBatchGenTextDataset,
    DataCollatorForSumBatchGenLanguageModeling,

)
import sys, os
sys.path.insert(1, '/u/scr/xlisali/contrast_LM/transformers/examples/control')
from train_control import PrefixTuning, PrefixEmbTuning


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


# def set_seed(args):
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.n_gpu > 0:
#         torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}

def read_e2e_files(path, tokenizer, lowdata_token=None):
    file_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src, tgt = line.strip().split('||')
            # URGENT CHANGE
            # src =  src + ' {}'.format(' summarize :')
            if lowdata_token is None:
                src = ' {} {}'.format(src, tokenizer.bos_token)
                # src =  src + ' {}'.format(tokenizer.bos_token)
            else:
                src = ' {} {} {}'.format(lowdata_token, src, tokenizer.bos_token)
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    return file_dict

def read_wp_files(path, tokenizer):
    file_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src, tgt = line.strip().split('|||')
            src = src + ' {}'.format(tokenizer.bos_token)
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    return file_dict


def read_classifySentiment_files(path, tokenizer):
    file_dict = []
    with open(path, 'r') as f:
        for line in f:
            tgt, src = line.strip().split('|||')
            src = src.replace("< br / >", "\n")
            src = ' {} {}'.format(src, tokenizer.bos_token)
            file_dict.append((src, tgt))
    return file_dict

def read_classifyTopic_files(path, tokenizer):
    file_dict = []
    with open(path, 'r') as f:
        for line in f:
            if (len(line) > 0 and not line.isspace()
                    and len(line.split('||')) == 2):
                tgt, src = line.strip().split('||')
            else:
                continue
            src = ' {} {}'.format(src, tokenizer.bos_token)
            file_dict.append((src, tgt))
    return file_dict


# def ids_to_text_without_prompt(tokenizer, generated_ids, prompt):
#     gen_text = tokenizer.batch_decode(
#         generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
#     )
#     for idx, text in enumerate(gen_text):
#         text_output = text[len(tokenizer.decode(prompt[idx], clean_up_tokenization_spaces=True)):]
#         idx = text_output.find(tokenizer.eos_token)
#     return lmap(str.strip, gen_text)

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

def ids_to_clean_text(tokenizer, generated_ids):
    gen_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return lmap(str.strip, gen_text)

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]

def flatten_list(summary_ids):
    return [x for x in itertools.chain.from_iterable(summary_ids)]

def calculate_rouge(output_lns, reference_lns, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

def test_epoch_end(outputs, prefix="test"):
    # losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
    # loss = losses["loss"]
    # print(loss)
    metric_names = ROUGE_KEYS
    generative_metrics = {
        k: np.array([x[k] for x in outputs]).mean() for k in metric_names + ["gen_time", "gen_len"]
    }
    # metric_val = (
    #     generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
    # )
    # metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
    # generative_metrics.update({k: v.item() for k, v in losses.items()})
    losses = {}
    losses.update(generative_metrics)
    all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
    preds = flatten_list([x["preds"] for x in outputs])
    return {
        "log": all_metrics,
        "preds": preds,
        # f"{prefix}_loss": loss,
        # f"{prefix}_{self.val_metric}": metric_tensor,
    }

def test_step(model, gpt2, batch, batch_idx, args, tokenizer, beam_handle, gold_handle):
    t0 = time.time()
    # TODO(LISA)
    # write the prompt generation from self.model.
    # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
    # get the prompt:
    bsz = batch["input_ids"].size(0)
    # prefix_prompt = model.get_prompt(bsz=bsz,)
    # expand to get bsz * sample_size.
    control_code = None
    print('control code is ', control_code)
    # prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=1)
    prompt = model.get_prompt(bsz=1)
    num_beamsize = 5
    prompt = [x.expand(-1, num_beamsize*bsz, -1, -1, -1) for x in prompt]


    # print('the max length of the model is {}'.format(model.config.max_length))

    input_ids = batch["input_ids"] #bsz, seqlen
    seqlen = len(input_ids[0])
    # bos_seq = torch.ones(bsz, 1).fill_(tokenizer.bos_token_id)
    input_attn  = batch["src_attn"].to(gpt2.device)
    prefix_attn = torch.ones(bsz, model.config.preseqlen).long().to(gpt2.device)
    input_attn = torch.cat([prefix_attn, input_attn], dim=-1)

    # input_ids = torch.cat([input_ids, bos_seq], dim=-1)
    # print(input_ids.shape)
    # print(input_ids.shape, input_attn.shape)

    # torch.set_printoptions(profile="full")
    # print(input_ids)
    # print(input_attn)
    # torch.set_printoptions(profile="default")
    # print(prompt[5][0][0][0])
    generated_ids = gpt2.generate(
        input_ids=input_ids.to(gpt2.device),
        emb_match=None,
        control_code=None,
        past_key_values=prompt,
        attention_mask=input_attn,
        use_prefix_test=True,
        max_length=args.length + seqlen, # what is self.eval_max_length
        min_length=5,
        temperature=args.temperature,
        top_k=args.k,
        top_p=0.9,  # top_p=0.5,
        repetition_penalty=args.repetition_penalty,  ##args.repetition_penalty,
        do_sample=False,
        num_beams=5,
        bad_words_ids=[[628], [198]] if True else None,
        num_return_sequences=1,
    )
    # clean up generated_ids
    bsz, seqlen = input_ids.shape
    generated_ids = generated_ids[:,seqlen:]
    # print(generated_ids)

    # generated_ids = gpt2.generate(
    #     batch["input_ids"],
    #     past_key_values=prefix_prompt,
    #     attention_mask=batch["attention_mask"],
    #     use_cache=True,
    #     use_prefix=True,
    #     decoder_start_token_id=self.decoder_start_token_id,
    #     num_beams=self.eval_beams,
    #     max_length=self.eval_max_length,
    # )
    gen_time = (time.time() - t0) / batch["input_ids"].shape[0]

    preds: List[str] = ids_to_clean_text(tokenizer, generated_ids)
    # src: List[str] = ids_to_clean_text(tokenizer, input_ids)
    # print(src)
    target: List[str] = ids_to_clean_text(tokenizer, batch["labels"])
    # print(preds)
    # print(target)
    # loss_tensors = self._step(batch)
    # base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
    # print('INPUT:', self.ids_to_clean_text(batch["input_ids"]))
    # print(preds, target)

    for predd in preds:
        print(predd, file=beam_handle)

    for tgtt in target:
        print(tgtt, file=gold_handle)

    base_metrics = {}
    rouge: Dict = calculate_rouge(preds, target)
    summ_len = np.mean(lmap(len, generated_ids))
    base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
    return base_metrics


def read_webnlg_files(path, tokenizer):
    file_dict = {}

    with open(path) as f:
        lines_dict = json.load(f)

    full_rela_lst = []
    full_src_lst = []
    # full_tgt_lst = []
    total_count = 0
    for i, example in enumerate(lines_dict['entries']):
        sents = example[str(i + 1)]['lexicalisations']
        triples = example[str(i + 1)]['modifiedtripleset']

        rela_lst = []
        temp_triples = ''
        for j, tripleset in enumerate(triples):
            subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
            rela_lst.append(rela)
            if i > 0:
                temp_triples += ' | '
            temp_triples += '{} : {} : {}'.format(subj, rela, obj)

        temp_triples = ' {} {}'.format(temp_triples, tokenizer.bos_token)


        for sent in sents:
            if True: #sent["comment"] == 'good'
                if (temp_triples,tuple(rela_lst)) not in file_dict:
                    file_dict[(temp_triples,tuple(rela_lst))] = []
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(tuple(rela_lst))
                file_dict[(temp_triples,tuple(rela_lst))].append(sent["lex"])


    print(len(file_dict), len(full_src_lst))
    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)

    return file_dict


def read_triples_files2(path, tokenizer):
    file_src = []
    file_tgt = []

    with open(path) as f:
        lines_dict = json.load(f)

    print(len(lines_dict))
    full_rela_lst = []
    full_src_lst = []
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

        temp_triples = ' {} {}'.format(temp_triples, tokenizer.bos_token)

        file_src.append((temp_triples, tuple(rela_lst)))
        # file_tgt

        for sent in example['annotations']:
            if (temp_triples, tuple(rela_lst)) not in file_dict:
                file_dict[(temp_triples, tuple(rela_lst))] = []
                full_src_lst.append(temp_triples)
                full_rela_lst.append(tuple(rela_lst))
            file_dict[(temp_triples, tuple(rela_lst))].append(sent['text'])

    print(len(file_dict), len(full_src_lst))
    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)
    return file_dict

def read_triples_files(path, tokenizer):
    file_dict = {}

    with open(path) as f:
        lines_dict = json.load(f)

    print(len(lines_dict))
    full_rela_lst = []
    full_src_lst = []
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

        temp_triples = ' {} {}'.format(temp_triples, tokenizer.bos_token)

        for sent in example['annotations']:
            if (temp_triples, tuple(rela_lst)) not in file_dict:
                file_dict[(temp_triples, tuple(rela_lst))] = []
                full_src_lst.append(temp_triples)
                full_rela_lst.append(tuple(rela_lst))
            file_dict[(temp_triples, tuple(rela_lst))].append(sent['text'])

    print(len(file_dict), len(full_src_lst))
    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)
    return file_dict

# def write_e2e_corr(prompt_lst, file_dict, corr_path):
#     with open(corr_path, 'w') as f:
#         for x in prompt_lst:
#             for line in file_dict[x]:
#                 print(line, file=f)
#             print('', file=f)
#     return

def write_e2e_corr(prompt_lst, file_dict, corr_path):
    print(len(prompt_lst))
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            for line in file_dict[x]:
                if not line.strip():
                    print('PROBLEM', line,'PROBLEM',file_dict[x] )
                else:
                    print(line, file=f)
            print('', file=f)

    # buf = [[]]
    # with open(corr_path, 'r') as fh:
    #     for line in fh:
    #         line = line.strip()
    #         if True:
    #             # print(line)
    #             if not line:
    #                 buf.append([])
    #             else:
    #                 buf[-1].append(line)
    #         else:
    #             buf.append(line)
    # if not buf[-1]:
    #     del buf[-1]
    #
    # print(buf[:3])
    #
    # print(len(buf))

    return

def write_e2e_src(prompt_lst, corr_path):
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            print(x, file=f)
    return



def get_emb(sent_lst, word_lst, num_layer=1):
    # load bert
    tokenizer_bert = BertTokenizerFast.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained('bert-large-uncased', return_dict=True).cuda()
    for param in model.parameters():
        param.requires_grad = False

    device = model.device

    edited_sent = []
    chosen_word = []
    with torch.no_grad():
        computed_ = 0
        mid_ = 300
        full_score = []
        while computed_ < len(sent_lst):
            temp_sent = sent_lst[computed_:computed_ + mid_]
            temp_word = word_lst[computed_:computed_ + mid_]
            temp_input = tokenizer_bert(temp_sent, return_tensors="pt", padding=True,
                                        is_split_into_words=False, return_offsets_mapping=True, add_special_tokens=True)
            input_ids = temp_input["input_ids"]
            # print(temp_input.keys())
            mask_input = temp_input['attention_mask']
            bsz, seqlen = input_ids.shape

            # print(input_ids.shape)

            cand_idx = tokenizer_bert(temp_word, add_special_tokens=False)['input_ids']
            # print(cand_idx)
            # if BPE has multiple subwords.
            cand_idx = torch.tensor([i[-1] for i in cand_idx])  # bsz
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
            word_temp = []
            for i, (sent1, word1) in enumerate(zip(temp_sent, temp_word)):
                # TODO: could check against the offests and make final changes!
                temp_idx1 = temp_input["offset_mapping"][i][mask_idx[i, 1]]
                # print(word1, sent1)
                # print(sent1[temp_idx1[0]:temp_idx1[1]])
                sent1 = sent1.split()
                widx = sent1.index(word1)
                by_tokenl = sum([len(l) + 1 for l in sent1[:widx]])
                by_tokenr = sum([len(l) + 1 for l in sent1[:widx + 1]]) - 1
                # print(by_tokenl, by_tokenr, temp_idx1)
                if by_tokenl != temp_idx1[0].item() and by_tokenr != temp_idx1[1].item():
                    # print('dangerous')
                    # print(sent1, word1, by_tokenl, by_tokenr, temp_idx1)
                    # simple option: delete it form input_ids
                    keep_mask.append(False)
                    continue
                else:
                    keep_mask.append(True)
                new_sent = [word1, '[BOS]'] + sent1[:widx] + ['[', sent1[widx], ']'] + sent1[widx + 1:] + ['[EOS]']
                assert len(new_sent) == len(sent1) + 5
                edit_temp.append(new_sent)
                word_temp.append(word1)

            keep_mask = torch.tensor(keep_mask)
            # print(keep_mask.shape, input_ids.shape, mask.shape, 'hi')
            input_ids = input_ids[keep_mask]
            mask = mask[keep_mask]
            mask_input = mask_input[keep_mask]
            # print(input_ids.shape, mask.shape, len(edit_temp))
            assert input_ids.size(0) == len(edit_temp)

            edited_sent += edit_temp
            chosen_word += word_temp
            # print(len(edited_sent), len(chosen_word))

            outputs = model(input_ids.to(device), attention_mask=mask_input.to(device), output_hidden_states=True)

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

    return full_score, edited_sent, chosen_word

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def read_doc_for_embmatch(file_name, num_layer):
    word_lst = []
    sent_lst = []
    with open(file_name, 'r') as f:
        for line in f:
            word, sent = line.strip().split('||')
            word_lst.append(word)
            sent_lst.append(sent)

    emb_match, sent_cleaned_lst, chosen_word = get_emb(sent_lst, word_lst, num_layer=num_layer)
    prompt_text_lst = [word + ' [BOS]' for word in chosen_word]
    return prompt_text_lst, emb_match.split(1), sent_cleaned_lst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained tokenizer or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--prefixModel_name_or_path",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained PrefixTuning Model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--task_mode", type=str, default="embMatch")
    parser.add_argument("--control_mode", type=str, default="yes")
    parser.add_argument("--prefix_mode", type=str, default="activation")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--gen_dir", type=str, default="e2e_results_conv")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--tuning_mode", type=str, default="finetune", help="prefixtune or finetune")
    parser.add_argument("--objective_mode", type=int, default=2)
    parser.add_argument("--format_mode", type=str, default="peek", help="peek, cat, nopeek, or infix")
    parser.add_argument("--optim_prefix", type=str, default="no", help="optim_prefix")
    parser.add_argument("--preseqlen", type=int, default=5, help="preseqlen")

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--control_dataless", type=str, default="no", help="control dataless mode")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args.seed)

    # Initialize the model and tokenizer

    if args.tuning_mode == 'finetune':
        print(args.tuning_mode, args.model_name_or_path)
        try:
            args.model_type = args.model_type.lower()
            model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        if args.model_name_or_path:
            print('loading the trained tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        elif args.tokenizer_name:
            print('loading from the init tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

        # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        print(config)
        model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
        model.to(args.device)

    elif args.tuning_mode == 'adaptertune':
        print(args.tuning_mode, args.model_name_or_path)

        try:
            args.model_type = args.model_type.lower()
            _, tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        if args.model_name_or_path:
            print('loading the trained tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        elif args.tokenizer_name:
            print('loading from the init tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        print(config)
        model = GPT2LMHeadModelAdapter.from_pretrained(
            args.model_name_or_path,
            config=config,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            cache_dir=args.cache_dir,
        )

        model.to(args.device)
        args.tuning_mode = 'finetune'

    elif args.tuning_mode == 'bothtune':
        print(args.tuning_mode, args.model_name_or_path, args.prefixModel_name_or_path)
        try:
            args.model_type = args.model_type.lower()
            model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        if args.prefixModel_name_or_path:
            print('loading the trained tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.prefixModel_name_or_path, cache_dir=args.cache_dir)
        elif args.tokenizer_name:
            print('loading from the init tokenizer')
            assert False, "should load from the prefixModel_name_or_path tokenizer"
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

            # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        print(config)
        model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
        model.to(args.device)
        gpt2 = model


        print('loading from PrefixTuning.', args.prefixModel_name_or_path, )
        if args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if args.prefixModel_name_or_path is not None:
            config = AutoConfig.from_pretrained(args.prefixModel_name_or_path, cache_dir=args.cache_dir)
            print(config)

            if args.prefix_mode == 'embedding':
                model = PrefixEmbTuning.from_pretrained(
                    args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in args.prefixModel_name_or_path, ),
                    config=config,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=args.preseqlen,
                    use_infix=(args.format_mode == 'infix')
                )

            elif args.prefix_mode == 'activation':

                model = PrefixTuning.from_pretrained(
                    args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in args.prefixModel_name_or_path, ),
                    config=config,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=args.preseqlen,
                    use_infix=(args.format_mode == 'infix')
                )

            model.to(args.device)




    elif args.tuning_mode == 'prefixtune':

        print('loading from PrefixTuning.', args.prefixModel_name_or_path,)
        if args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            assert False, 'shouldn not init config from scratch. '
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        try:
            args.model_type = args.model_type.lower()
            model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        if args.model_name_or_path:
            print('loading the trained tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        elif args.tokenizer_name:
            print('loading from the init tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

        # TODAYFIX.
        config._my_arg_tune_mode = args.tuning_mode
        config._my_arg_task_mode = args.task_mode
        config._objective_mode = args.objective_mode
        model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
        model.to(args.device)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        # TODO LISA
        add_pad = False

        if args.model_name_or_path == 'gpt2-medium':
            if args.task_mode == 'dataless':
                print(args.tuning_mode, 'dataless setting, so no new tokens at all.')
                print('We do not add special tokens to the tokenizer, instead, we just finetune on <|endoftext|>')

                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

            elif add_pad:
                print('extending the size of word embeddings. to include the [PAD] ')
                num_added_tokens = tokenizer.add_special_tokens(
                    {'pad_token': '[PAD]'})
                embedding_layer = model.resize_token_embeddings(len(tokenizer))
            else:
                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)


            ########################################3

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)


        gpt2 = model

        # config._my_arg_task_mode = args.task_mode
        # config._my_arg_control = True
        # config.train_weights = 'no'
        print(config)
        if args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if args.prefixModel_name_or_path is not None:

            #################
            #
            config = AutoConfig.from_pretrained(args.prefixModel_name_or_path, cache_dir=args.cache_dir )
            print(config)

            if args.prefix_mode == 'embedding':
                model = PrefixEmbTuning.from_pretrained(
                    args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in args.prefixModel_name_or_path, ),
                    config=config,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=args.preseqlen,
                    use_infix=(args.format_mode == 'infix')
                )

            elif args.prefix_mode == 'activation':

                model = PrefixTuning.from_pretrained(
                    args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in args.prefixModel_name_or_path, ),
                    config=config,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=args.preseqlen,
                    use_infix=(args.format_mode == 'infix')
                )
            #
            ######################

            # model = PrefixTuning.from_pretrained(
            #     args.prefixModel_name_or_path,
            #     from_tf=bool(".ckpt" in args.prefixModel_name_or_path,),
            #     config=config,
            #     model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=args.preseqlen,
            # )
            model.to(args.device)

            # print('-'*100)
            # print(model.training)
            # print(gpt2.training)
            # model.train()
            # gpt2.train()
            # print(model.training)
            # print(gpt2.training)
            # model.eval()
            # gpt2.eval()
            # print(model.training)
            # print(gpt2.training)
            # print('-' * 100)

        else:
            assert False, "prefixModel_name_or_path is NONE."



    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    if args.task_mode == 'data2text':

        QUICK_CHECK = False

        if QUICK_CHECK:

            prompt_text_lst = [
                "name : Blue Spice | Type : coffee shop | area : city centre {}".format(tokenizer.bos_token),
                "name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 {}".format(tokenizer.bos_token),
                "name : Blue Spice | Type : pub | food : Chinese | area : city centre | family friendly : no {}".format(tokenizer.bos_token),
                "name : Blue Spice | Type : restaurant | food : Chinese | area : city centre | family friendly : yes | near : Rainbow Vegetarian Café {}".format(tokenizer.bos_token),
                "name : Giraffe | Type : restaurant | food : Fast food | area : riverside | family friendly : no | near : Rainbow Vegetarian Café {}".format(tokenizer.bos_token),
                "name : The Cricketers | Type : coffee shop | customer rating : 1 out of 5 | family friendly : yes | near : Avalon {}".format(tokenizer.bos_token),
                "name : The Cricketers | Type : restaurant | food : Chinese | price : high | customer rating : 1 out of 5 | area : city centre | family friendly : no {}".format(tokenizer.bos_token),
                "name : The Mill | Type : restaurant | food : English | price : moderate | area : riverside | family friendly : yes | near : Raja Indian Cuisine {}".format(tokenizer.bos_token),

            ]
            decode_mode = 'beam'

        else:
            # TODO.LISA
            # test_path = '/u/scr/xlisali/e2e_data/contain_near_Type_src1_test.txt'
            if ('lowdata' in args.model_name_or_path) or (args.prefixModel_name_or_path is not None and 'lowdata' in args.prefixModel_name_or_path):
                test_path = '/u/scr/xlisali/e2e_data/src1_valid.txt'
            else:
                test_path = '/u/scr/xlisali/e2e_data/src1_test.txt'

            print('using the test path ', test_path)
            # test_path = '/u/scr/xlisali/e2e_data/src1_valid.txt'
            if args.prefixModel_name_or_path is not None:
                temp = os.path.basename(args.prefixModel_name_or_path)
            else:
                temp = os.path.basename(args.model_name_or_path)

            if 'lowdata' in temp and 'finetune' in temp:
                lowdata_token = temp.split('_t=')[1].split('-checkpoint-')[0]
                print('the LOWDATA token is {}'.format(lowdata_token))
            else:
                lowdata_token = None
            prompt_text_dict = read_e2e_files(test_path, tokenizer, lowdata_token)

            # print(prompt_text_dict)
            prompt_text_lst = list(prompt_text_dict.keys())
            split_file = 'valid'
            decode_mode = 'beam'
            curr_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file, decode_mode))
            print(curr_dir)
            gold_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file,'gold'))
            print(gold_dir)
            write_e2e_corr(prompt_text_lst, prompt_text_dict, gold_dir)
            src_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                   args.gen_dir,
                                   '{}_{}_{}'.format(temp,split_file, 'src'))
            write_e2e_src(prompt_text_lst, src_dir)
            out_handle = open(curr_dir, 'w')


    elif args.task_mode == 'webnlg' or args.task_mode == 'triples':
        QUICK_CHECK = False
        if args.task_mode == 'webnlg':
            # test_path = "/u/scr/xlisali/WebNLG/webnlg-dataset/release_v2/json/webnlg_release_v2_test.json"
            test_path = "/u/scr/xlisali/WebNLG/webnlg-dataset/webnlg_challenge_2017/test.json"
            prompt_text_dict = read_webnlg_files(test_path, tokenizer)
        elif args.task_mode == 'triples':
            test_path = "/u/scr/xlisali/DART/dart/data/v1.1.1/dart-v1.1.1-full-test.json"
            prompt_text_dict = read_triples_files(test_path, tokenizer)

        if QUICK_CHECK:
            prompt_text_pair = list(prompt_text_dict.keys())[:20]
            prompt_text_lst, prompt_rela_lst = zip(*prompt_text_pair)
            decode_mode = 'beam'

        else:
            prompt_text_pair = list(prompt_text_dict.keys())
            prompt_text_lst, prompt_rela_lst = zip(*prompt_text_pair)
            if args.prefixModel_name_or_path is not None:
                temp = os.path.basename(args.prefixModel_name_or_path)
            else:
                temp = os.path.basename(args.model_name_or_path)
            # print(prompt_text_dict)
            split_file = 'test' # test
            decode_mode = 'beam'
            curr_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file, decode_mode))

            print(curr_dir)
            gold_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file, 'gold'))

            print(gold_dir)
            write_e2e_corr(prompt_text_pair, prompt_text_dict, gold_dir)
            src_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file, 'src'))

            write_e2e_src(prompt_text_pair, src_dir)


            out_handle = open(curr_dir, 'w')

    elif args.task_mode == 'writingPrompts':
        QUICK_CHECK = True
        test_path = "/juice/u/xlisali/WritingPrompts/writingPrompts/test_small.txt"
        prompt_text_dict = read_wp_files(test_path, tokenizer)
        args.num_return_sequences = 1

        if QUICK_CHECK:
            prompt_text_lst = list(prompt_text_dict.keys())[:20]
            print(prompt_text_lst)
            decode_mode = 'nucleus'

        else:
            prompt_text_pair = list(prompt_text_dict.keys())
            prompt_text_lst, prompt_rela_lst = zip(*prompt_text_pair)
            if args.prefixModel_name_or_path is not None:
                temp = os.path.basename(args.prefixModel_name_or_path)
            else:
                temp = os.path.basename(args.model_name_or_path)
            # print(prompt_text_dict)
            split_file = 'test' # test
            decode_mode = 'beam'
            curr_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file, decode_mode))

            print(curr_dir)
            gold_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file, 'gold'))

            print(gold_dir)
            write_e2e_corr(prompt_text_pair, prompt_text_dict, gold_dir)

            src_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                   args.gen_dir,
                                   '{}_{}_{}'.format(temp, split_file, 'src'))

            write_e2e_src(prompt_text_pair, src_dir)
            out_handle = open(curr_dir, 'w')


    elif args.task_mode == 'sentiment' or args.task_mode == 'topic':
        QUICK_CHECK = False
        args.num_return_sequences = 3

        if QUICK_CHECK:
            prompt_text_lst = [" positive {}".format(tokenizer.bos_token)] * 10  + [" negative {}".format(tokenizer.bos_token)] * 10
            print(prompt_text_lst)
            decode_mode = 'nucleus'

        else:
            #UNCHECKED
            topic_prompt_pplm_lst = ['In summary', 'This essay discusses', 'Views on', 'The connection',
                               'Foundational to this is', 'To review', 'In brief', 'An illustration of', 'Furthermore',
                               'The central theme', 'To conclude', 'The key aspect', 'Prior to this', 'Emphasised are',
                               'To summarize', 'The relationship', 'More importantly', 'It has been shown',
                               'The issue focused on', 'In this essay']

            sent_prompt_pplm_lst = ['Once upon a time', 'The book', 'The chicken', 'The city', 'The country', 'The horse',
                               'The lake', 'The last time']

            if args.task_mode == 'topic':
                pplm_lst = topic_prompt_pplm_lst
                prompt_text_lst = []
                for i in range(len(pplm_lst)):
                    prompt_text_lst.append(" business {} {}".format(tokenizer.bos_token, pplm_lst[i]))
                    prompt_text_lst.append(" sports {} {}".format(tokenizer.bos_token, pplm_lst[i]))
                    prompt_text_lst.append(" science {} {}".format(tokenizer.bos_token, pplm_lst[i]))
                    prompt_text_lst.append(" world {} {}".format(tokenizer.bos_token, pplm_lst[i]))
            else:
                pplm_lst = sent_prompt_pplm_lst
                prompt_text_lst = []
                for i in range(len(pplm_lst)):
                    prompt_text_lst.append(" positive {} {}".format(tokenizer.bos_token, pplm_lst[i]))
                    prompt_text_lst.append(" negative {} {}".format(tokenizer.bos_token, pplm_lst[i]))

            if args.prefixModel_name_or_path is not None:
                temp = os.path.basename(args.prefixModel_name_or_path)
            else:
                temp = os.path.basename(args.model_name_or_path)
            split_file = 'test' # test
            decode_mode = 'nucleus'

            curr_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file, decode_mode))
            print(curr_dir)

            src_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                   args.gen_dir,
                                   '{}_{}_{}'.format(temp, split_file, 'src'))


            write_e2e_src(prompt_text_lst, src_dir)
            out_handle = open(curr_dir, 'w')


    elif args.task_mode == 'classify-sentiment' or args.task_mode == 'classify-topic':
        QUICK_CHECK = False
        if args.task_mode == 'classify-sentiment':
            test_path = "/u/scr/xlisali/IMDB/test.txt"
            prompt_text_dict = read_classifySentiment_files(test_path, tokenizer)
        elif args.task_mode == 'classify-topic':
            test_path = "/u/scr/xlisali/contrast_LM/transformers/examples/text-classification/glue_data/AG-news/dev1.tsv"
            prompt_text_dict = read_classifyTopic_files(test_path, tokenizer)

        args.num_return_sequences = 1

        if QUICK_CHECK:
            prompt_text_lst, prompt_text_tgt = zip(*prompt_text_dict)
            prompt_text_lst = prompt_text_lst[:20]
            print(prompt_text_lst)
            decode_mode = 'greedy'

        else:
            #UNCHECKED
            prompt_text_lst, prompt_text_tgt = zip(*prompt_text_dict)
            if args.prefixModel_name_or_path is not None:
                temp = os.path.basename(args.prefixModel_name_or_path)
            else:
                temp = os.path.basename(args.model_name_or_path)
            # print(prompt_text_dict)
            split_file = 'test' # test
            decode_mode = 'greedy'
            curr_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file, decode_mode))

            print(curr_dir)
            gold_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                    args.gen_dir,
                                    '{}_{}_{}'.format(temp, split_file, 'gold'))

            print(gold_dir)
            write_e2e_src(prompt_text_tgt, gold_dir)
            src_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                   args.gen_dir,
                                   '{}_{}_{}'.format(temp, split_file, 'src'))

            write_e2e_src(prompt_text_lst, src_dir)
            out_handle = open(curr_dir, 'w')

            print('the total length of generation should be {}'.format(len(prompt_text_lst)))




    elif args.task_mode == 'cnndm' or args.task_mode =='xsum':
        QUICK_CHECK = False
        if args.task_mode == 'cnndm':
            # test_path = "/u/scr/xlisali/WebNLG/webnlg-dataset/release_v2/json/webnlg_release_v2_test.json"
            test_path = "/u/scr/xlisali/contrast_LM/transformers/examples/seq2seq/cnn_dm/test.source"
            max_source_length = 512
            max_target_length = 142
            args.length = max_target_length
            # prompt_text_dict = read_sum_files(test_path, tokenizer, max_source_len, max_target_len)
        elif args.task_mode == 'xsum':
            test_path = "/u/scr/xlisali/contrast_LM/transformers/examples/seq2seq/xsum/test.source"
            max_source_length = 512
            max_target_length = 100
            args.length = max_target_length
            # prompt_text_dict = read_sum_files(test_path, tokenizer, max_source_len, max_target_len)

        tokenizer.padding_side = "left"

        dataset = LineByLineSumBatchGenTextDataset(tokenizer=tokenizer, file_path=test_path,
                                           block_size=1024, bos_tok=tokenizer.bos_token,
                                           eos_tok=tokenizer.eos_token, max_source_length=max_source_length,
                                           max_target_length=max_target_length, )


        data_collator = DataCollatorForSumBatchGenLanguageModeling(
            tokenizer=tokenizer, mlm=False, mlm_probability=0.0,max_source_length=max_source_length,
            max_target_length=max_target_length,
        )

        # prompt_text_pair = list(prompt_text_dict.keys())
        # prompt_text_lst, prompt_rela_lst = zip(*prompt_text_pair)
        if args.prefixModel_name_or_path is not None:
            temp = os.path.basename(args.prefixModel_name_or_path)
        else:
            temp = os.path.basename(args.model_name_or_path)
        # # print(prompt_text_dict)
        split_file = 'test'  # test
        decode_mode = 'beam'
        curr_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                args.gen_dir,
                                '{}_{}_{}_batch'.format(temp, split_file, decode_mode))
        #
        # print(curr_dir)
        gold_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
                                args.gen_dir,
                                '{}_{}_{}_batch'.format(temp, split_file, 'gold'))
        #
        # print(gold_dir)
        # write_e2e_corr(prompt_text_pair, prompt_text_dict, gold_dir)
        # src_dir = os.path.join('/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/',
        #                        args.gen_dir,
        #                        '{}_{}_{}'.format(temp, split_file, 'src'))
        #
        # write_e2e_src(prompt_text_pair, src_dir)
        #
        out_handle_beam = open(curr_dir, 'w')
        out_handle_gold = open(gold_dir, 'w')



    if args.control_mode == 'yes':
        print('processing control codes')


    # Since we are doing batch processing, should use data loader and batch it, rather than using these for-loops.
    data_loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    collate_fn=data_collator,
                    shuffle=False,
                    num_workers=4,
                    sampler=None,
                )

    out_lst = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # print(batch)
            # batch = model.transfer_batch_to_device(batch, model.device)
            print(batch_idx)
            # if batch_idx >= 5:
            #     break
            # print(batch['input_ids'].device, model.device)
            out = test_step(model, gpt2, batch, batch_idx, args, tokenizer, beam_handle=out_handle_beam, gold_handle=out_handle_gold)
            out_lst.append(out)
            for x in out['preds']:
                print(x)
            # batch = model.transfer_batch_to_device(batch, 'cpu')
        result = test_epoch_end(out_lst)

    out_handle_beam.close()
    out_handle_gold.close()

    print('writing the test results to ', curr_dir)
    print('writing the gold results to ', gold_dir)


    # print(result)
    for k, v in result.items():
        if k != 'preds':
            print(k, v)


    return











































    for prompt_idx, prompt_text in enumerate(prompt_text_lst):

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt


        if args.control_mode == 'yes' and args.control_dataless != 'yes':
            # URGENT, check whether the next line is necessary?
            # control_code = torch.LongTensor(control_codes[prompt_idx]).to(model.device).unsqueeze(0).expand(args.num_return_sequences, -1)
            control_code = None
            pass
            # print(control_code)
            # print(control_code.shape)
            # print(input_ids)
            # DEBUG
            # control_code = None
        else:
            control_code = None
        # for param in model.base_model.parameters():
        #     print(param.requires_grad)

        # if args.control_dataless == 'yes':
        if args.tuning_mode == 'prefixtune' or args.tuning_mode == 'bothtune':
            if args.task_mode == 'embMatch':
                control_code = emb_match[prompt_idx].to(model.device)
            elif args.task_mode == 'keyword':
                control_code = torch.LongTensor(control_codes[prompt_idx]).to(model.device).unsqueeze(0)
                print(control_code)
            elif args.task_mode == 'topic_old':
                control_code = torch.LongTensor(control_codes[prompt_idx]).to(model.device).unsqueeze(0)
                print(control_code)
            elif args.task_mode == 'data2text':
                src = prompt_text_lst[prompt_idx].split()[:-1] # remove the bos token.
                # print(src)
                src = ' '.join(src)
                catl = src.split('|')
                cat = [cc.split(':')[0].strip() for cc in catl]
                # print(cat)

                src_cat = tokenizer(cat, add_special_tokens=True, truncation=True, is_split_into_words=True)['input_ids']

                if args.format_mode == 'infix':
                    src = ' {} '.format(src)
                src = tokenizer(src, add_special_tokens=True, truncation=True, is_split_into_words=False)['input_ids']

                mode = None
                if 'cat2' in args.prefixModel_name_or_path or 'cat' in args.prefixModel_name_or_path:
                    mode = 'cat'
                elif 'nopeek' in args.prefixModel_name_or_path or 'nop' in args.prefixModel_name_or_path:
                    mode = 'nopeek'
                elif 'peek' in args.prefixModel_name_or_path or 'pee' in args.prefixModel_name_or_path:
                    mode = 'peek'
                elif 'prefixtune15' in args.prefixModel_name_or_path:
                    mode = 'instruction_based'
                    # assert False, "prefixtune20 shouldn't be processed here."
                else:
                    if args.format_mode == 'infix':
                        mode = 'infix'
                    else:
                        assert False, "Given that it's in prefix tuning mode, need to specify a valid prefix mode, " \
                                      "(cat, nopeek, peek)"

                print(mode)

                if mode == 'cat':
                    cc = src_cat
                elif mode == 'peek' or mode == 'nopeek':
                    cc = src
                elif mode == 'infix':
                    cc = src
                # print('control code is ', cc)

                if mode == 'nopeek' or mode == 'infix':
                    # print('the old input_ids is ', input_ids) # this looks right
                    input_pp = tokenizer.bos_token
                    encoded_prompt = tokenizer(input_pp, add_special_tokens=True, truncation=True, return_tensors="pt", is_split_into_words=False)['input_ids'].to(model.device)
                    input_ids = encoded_prompt

                if mode in ['cat', 'peek', 'nopeek', 'infix']:
                    control_code = torch.LongTensor(cc).to(model.device).unsqueeze(0)
                elif mode == 'instruction_based':
                    control_code = None
                else:
                    assert False, "invalid mode type."

                # TODO.LISA
                if config.optim_prefix:
                    control_code = None
            elif args.task_mode == 'webnlg' or args.task_mode == 'triples':
                src = prompt_text_lst[prompt_idx].split()[:-1]
                print(src)
                src = ' '.join(src)
                cat = prompt_rela_lst[prompt_idx]
                print(cat)
                src_cat = tokenizer(cat, add_special_tokens=True, truncation=True, is_split_into_words=True)['input_ids']
                src = tokenizer(src, add_special_tokens=True, truncation=True, is_split_into_words=False)['input_ids']

                mode = None
                if 'cat2' in args.prefixModel_name_or_path or 'cat' in args.prefixModel_name_or_path:
                    mode = 'cat'
                elif 'nopeek' in args.prefixModel_name_or_path or 'nop' in args.prefixModel_name_or_path:
                    mode = 'nopeek'
                elif 'peek' in args.prefixModel_name_or_path or 'pee' in args.prefixModel_name_or_path:
                    mode = 'peek'
                elif 'tune_y_' in args.prefixModel_name_or_path or config.optim_prefix:
                    mode = 'instruction_based'
                    # assert False, "prefixtune20 shouldn't be processed here."
                else:
                    if args.format_mode == 'infix':
                        mode = 'infix'
                    else:
                        assert False, "Given that it's in prefix tuning mode, need to specify a valid prefix mode, " \
                                      "(cat, nopeek, peek)"

                print(mode)

                if mode == 'cat':
                    cc = src_cat
                elif mode == 'peek' or mode == 'nopeek':
                    cc = src
                elif mode == 'infix':
                    cc = src
                # print('control code is ', cc)

                if mode == 'nopeek' or mode == 'infix':
                    input_pp = tokenizer.bos_token
                    encoded_prompt = tokenizer(input_pp, add_special_tokens=True, truncation=True, return_tensors="pt", is_split_into_words=False)['input_ids'].to(model.device)
                    input_ids = encoded_prompt

                if mode in ['cat', 'peek', 'nopeek', 'infix']:
                    control_code = torch.LongTensor(cc).to(model.device).unsqueeze(0)
                elif mode == 'instruction_based':
                    control_code = None
                else:
                    assert False, "invalid mode type."

                # TODO.LISA
                if config.optim_prefix:
                    control_code = None

            elif args.task_mode == 'writingPrompts':
                # currently only trained with optim_prefix=yes
                src = prompt_text_lst[prompt_idx].split()[:-1]
                src = ' '.join(src)
                print(src)
                src = tokenizer(src, add_special_tokens=True, truncation=True, is_split_into_words=False)['input_ids']

                mode = None
                if 'cat2' in args.prefixModel_name_or_path or 'cat' in args.prefixModel_name_or_path:
                    mode = 'peek'
                elif 'nopeek' in args.prefixModel_name_or_path or 'nop' in args.prefixModel_name_or_path:
                    mode = 'nopeek'
                elif 'peek' in args.prefixModel_name_or_path or 'pee' in args.prefixModel_name_or_path:
                    mode = 'peek'
                elif 'tune_y_' in args.prefixModel_name_or_path or config.optim_prefix:
                    mode = 'instruction_based'
                    # assert False, "prefixtune20 shouldn't be processed here."
                else:
                    if args.format_mode == 'infix':
                        mode = 'infix'
                    else:
                        assert False, "Given that it's in prefix tuning mode, need to specify a valid prefix mode, " \
                                      "(cat, nopeek, peek)"

                print(mode)

                if mode == 'peek' or mode == 'nopeek':
                    cc = src
                elif mode == 'infix':
                    cc = src
                # print('control code is ', cc)

                if mode == 'nopeek' or mode == 'infix' or args.format_mode == 'infix': #LISA
                    print('the old input_ids is ', input_ids)
                    input_pp = tokenizer.bos_token
                    encoded_prompt = tokenizer(input_pp, add_special_tokens=True, truncation=True, return_tensors="pt",
                                               is_split_into_words=False)['input_ids'].to(model.device)
                    input_ids = encoded_prompt

                if mode in ['cat', 'peek', 'nopeek', 'infix']:
                    print(cc)
                    control_code = torch.LongTensor(cc).to(model.device).unsqueeze(0)
                elif mode == 'instruction_based':
                    control_code = None
                else:
                    assert False, "invalid mode type."

                # TODO.LISA
                if config.optim_prefix:
                    control_code = None
                    assert control_code is None

            else:
                control_code = None
                print('control code is None')

            if args.format_mode != 'infix':
                print(config.optim_prefix, optim_prefix_bool)
                print('control code is ', control_code)
                prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=1)
            else:
                print(control_code)
                print(src)
                src = torch.LongTensor(src).to(model.device).unsqueeze(0)
                print(input_ids)
                prompt = model.get_prompt(src, None, gpt2=gpt2, bsz=1) #src, control_code=None, gpt2=None, bsz=None, attn_mask=None


            # if args.task_mode == 'writingPrompts':
            #     prompt = None
            # else:
            # print(args.num_return_sequences)
            prompt = [x.expand(-1, args.num_return_sequences , -1, -1, -1) for x in prompt]
            # print(prompt[0].shape)
            # print(input_ids.shape)

            # assert emb_match_temp is None
            # assert control_code is None
            print(decode_mode)
            if decode_mode == 'nucleus':
                output_sequences = gpt2.generate(
                    input_ids=input_ids,
                    emb_match=None,
                    control_code=None,
                    past_key_values=prompt,
                    max_length=args.length + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=0.8,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                )
            elif decode_mode == 'beam':
                output_sequences = gpt2.generate(
                    input_ids=input_ids,
                    emb_match=None,
                    control_code=None,
                    past_key_values=prompt,
                    max_length=args.length + len(encoded_prompt[0]),
                    min_length=5,
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=0.9, #top_p=0.5,
                    repetition_penalty=args.repetition_penalty, ##args.repetition_penalty,
                    do_sample=False,
                    num_beams=5,
                    bad_words_ids=[[628], [198]] if True else None,
                    num_return_sequences=1,
                )
                print(output_sequences)

            elif decode_mode == 'greedy':
                output_sequences = gpt2.generate(
                    input_ids=input_ids,
                    emb_match=None,
                    control_code=None,
                    past_key_values=prompt,
                    max_length=args.length + len(encoded_prompt[0]),
                    min_length=5,
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=0.5,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=False,
                    bad_words_ids=[[628], [198]] if True else None,
                    num_return_sequences=1,
                )

        elif args.tuning_mode == 'finetune':
            print(decode_mode)
            if decode_mode == 'nucleus':
                output_sequences = model.generate(
                    input_ids=input_ids,
                    emb_match=emb_match_temp,
                    control_code=control_code,
                    max_length=args.length + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=0.8,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                )
            elif decode_mode == 'beam':
                output_sequences = model.generate(
                    input_ids=input_ids,
                    emb_match=emb_match_temp,
                    control_code=control_code,
                    max_length=args.length + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=0.9, #0.5
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=False,
                    num_beams=5,
                    bad_words_ids=[[628], [198]] if True else None,
                    num_return_sequences=1,
                )

            elif decode_mode == 'greedy':
                output_sequences = model.generate(
                    input_ids=input_ids,
                    emb_match=emb_match_temp,
                    control_code=control_code,
                    max_length=args.length + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=0.5,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=False,
                    bad_words_ids=[[628], [198]] if True else None,
                    num_return_sequences=1,
                )


            # output_sequences = model.generate(
            #     input_ids=input_ids,
            #     emb_match=emb_match_temp,
            #     control_code=control_code,
            #     max_length=args.length + len(encoded_prompt[0]),
            #     temperature=args.temperature,
            #     top_k=args.k,
            #     top_p=args.p,
            #     repetition_penalty=args.repetition_penalty,
            #     do_sample=True,
            #     num_return_sequences=args.num_return_sequences,
            # )




        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        if QUICK_CHECK:
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                # args.stop_token = tokenizer.eos_token
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]

                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (
                    prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                )

                generated_sequences.append(total_sequence)
                print(total_sequence)
        else:
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                # args.stop_token = tokenizer.eos_token
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                print(text)
                text_output = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                idx = text_output.find(tokenizer.eos_token)
                if idx >= 0:
                    text_output = text_output[:idx]
                text_output = text_output.strip()

                if args.task_mode == 'topic' or args.task_mode == 'sentiment':
                    text_output = prompt_text + ' ' + text_output + ' [SPECIAL_END]'



                if text_output:
                    print(text_output, file=out_handle)
                else:
                    print('Error', file=out_handle)



                # # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                # total_sequence = (
                #         prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                # )
                #
                # generated_sequences.append(total_sequence)
                # print(total_sequence)



        print()

    # return generated_sequences

    if not QUICK_CHECK:
        out_handle.close()

    if args.task_mode == 'data2text':
        out_file_eval = curr_dir+'_eval'
        print(out_file_eval, '\n', gold_dir, '\n', curr_dir)
        os.system("python /u/scr/xlisali/e2e-metrics/measure_scores.py "
                  "{} {} -p  -t -H >> {}".format(gold_dir, curr_dir, out_file_eval))

    elif 'classify' in curr_dir:
        print(curr_dir)
        print(gold_dir)
        temp_command = "python /u/scr/xlisali/classify-eval/eval.py {} {}".format(gold_dir, curr_dir)
        print(temp_command)
        os.system(temp_command)

    if args.task_mode == 'topic' or args.task_mode == 'sentiment':
        print('view results at ')
        print(curr_dir)

        temp_command = "python /u/scr/xlisali/attribute-eval/eval.py {} ".format(curr_dir)
        print(temp_command)
        os.system(temp_command)



if __name__ == "__main__":
    main()


