
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
from transformers import (
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
)
import sys, os
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
def read_sum_files(path, tokenizer, max_source_length, max_target_length):
    print('106::read_sum_files(path, tokenizer, max_source_length, max_target_length)')
    src_file = path
    tgt_file = path[:-2] + 'de'
    print('108::path=', path)
    # file_dict = {}
    file_dict = []
    src_lines = []
    with open(src_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and not line.isspace():
                src_lines.append(line)
    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    for src, tgt in zip(src_lines, tgt_lines):
        # print('121::src=', src)
        src_bpe = tokenizer.encode(
            src, add_special_tokens=False, truncation=True, max_length=max_source_length,
            is_split_into_words=False
        )
        # print('126::src_bpe=', src_bpe)
        src_full = src_bpe + [tokenizer.bos_token_id] # add the bos token.
       
        src_full = tuple(src_full)
        # if src_full not in file_dict:
        #     file_dict[src_full] = [tgt]
        # else:
        #     print('should not happen')
        #     file_dict[src_full].append(tgt)
        
        file_dict.append(src_full)
    return file_dict

def write_e2e_corr(prompt_lst, file_dict, corr_path):
    print("352::len(prompt_lst)=", len(prompt_lst))
    print("353::type(file_dict)=", type(file_dict))
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            print('356::x=',x)
            print("type(x)=", type(x))
            for line in file_dict[x]:
                if not line.strip():
                    print('PROBLEM', line,'PROBLEM',file_dict[x] )
                else:
                    print(line, file=f)
            print('', file=f)
    return

def write_e2e_src(prompt_lst, corr_path):
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            print(x, file=f)
    return

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

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
    parser.add_argument("--gen_dir", type=str, default="")
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
    parser.add_argument("--tuning_mode", type=str, default="finetune", help="prefixtune or finetune")
    parser.add_argument("--eval_dataset", type=str, default="val", help="val or test")
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

    if args.tuning_mode == 'prefixtune':
        print('248::loading from PrefixTuning.', args.prefixModel_name_or_path,)
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
            print('261::loading the trained tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        elif args.tokenizer_name:
            print('264::loading from the init tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

        config._my_arg_tune_mode = args.tuning_mode
        config._my_arg_task_mode = args.task_mode
        config._objective_mode = args.objective_mode
        print("270::args.model_name_or_path=", args.model_name_or_path)
        print("271::args.cache_dir=", args.cache_dir)

        model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
        model.to(args.device)
        print('275::len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token=', len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        add_pad = False
        if args.model_name_or_path == 'gpt2-medium':
            if args.task_mode == 'dataless':
                print(args.tuning_mode, 'dataless setting, so no new tokens at all.')
                print('281:：We do not add special tokens to the tokenizer, instead, we just finetune on <|endoftext|>')
                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)
            elif add_pad:
                print('288:：extending the size of word embeddings. to include the [PAD] ')
                num_added_tokens = tokenizer.add_special_tokens(
                    {'pad_token': '[PAD]'})
                embedding_layer = model.resize_token_embeddings(len(tokenizer))
            else:
                print('293:：tokenizer.eos_token_id', tokenizer.eos_token_id)
                print('tokenizer.eos_token', tokenizer.eos_token)
                print('tokenizer.pad_token_id', tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print('tokenizer.pad_token, tokenizer.pad_token_id=', tokenizer.pad_token, tokenizer.pad_token_id)

        print('299:：len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token=', len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        gpt2 = model

        print('303::config', config)
        if args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"
        if args.prefixModel_name_or_path is not None:
            config = AutoConfig.from_pretrained(args.prefixModel_name_or_path, cache_dir=args.cache_dir )
            print('312::config=', config)
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
        else:
            assert False, "prefixModel_name_or_path is NONE."


    if args.fp16:
        model.half()
    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)
    
    if args.task_mode == 'nct': 
        # test_path = "/mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/data/ctorig/test_en-de.en"
        test_path = "/mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/test_en-de_context-2.en"
        # test_path = "/mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/test_de_gen_context-3.en"
        max_source_length = 512
        max_target_length = 512
        prompt_text_dict = read_sum_files(test_path, tokenizer, max_source_length, max_target_length)
        # print('prompt_text_dict=', prompt_text_dict)
        print('347::len(prompt_text_dict)=', len(prompt_text_dict))
        
        
        prompt_text_lst = list(prompt_text_dict)
        if args.prefixModel_name_or_path is not None:
            temp = os.path.basename(args.prefixModel_name_or_path)
        else:
            temp = os.path.basename(args.model_name_or_path)
        # print(prompt_text_dict)
        split_file = 'test' # test
        decode_mode = 'beam'
        if not os.path.exists('text-generation/'):
            os.mkdir('text-generation/')
        temp_dir = os.path.join('text-generation/', args.gen_dir)
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        curr_dir = os.path.join('text-generation/',
                                args.gen_dir,
                                '{}_{}_{}'.format(temp, split_file, decode_mode))

        print('367::curr_dir=', curr_dir)

        out_handle = open(curr_dir, 'w')

    load_path = "/mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/test_en-de_context-2_with-tgt.en"
    all_lines = []
    with open(load_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and not line.isspace():
                all_lines.append(line)
    num = len(all_lines)

    wfile = open("/mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/log.txt", "w")
    cnt = 0
    sentence_bar = '\t'
    translate_bar = ' = '
    form_2 = ""
    form_1 = ""
    for i in range(num):
        if all_lines[i] == '@@':
            cnt = 0
            form_2 = ""
            form_1 = ""
        else:
            cur_line = all_lines[i]
            cnt += 1
            if cur_line[:5] == 'agent':
                input_text = ""
                if cnt == 1:
                    input_text = cur_line + translate_bar
                elif cnt == 2:
                    input_text = form_1 + sentence_bar + cur_line + translate_bar
                else:
                    input_text = form_2 + sentence_bar + form_1 + sentence_bar + cur_line + translate_bar
                
                wfile.write(input_text + '\n')


                input_bpe = tokenizer.encode(input_text, add_special_tokens=False, truncation=True, max_length=512, is_split_into_words=False)
                input_full = input_bpe + [tokenizer.bos_token_id]

                #begin of gen

                if args.task_mode == 'nct':
                    # already processed
                    encoded_prompt = torch.LongTensor(input_full).unsqueeze(0)
                    print('414::encoded_prompt.shape=', encoded_prompt.shape)

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
                else:
                    control_code = None

                if args.tuning_mode == 'prefixtune' or args.tuning_mode == 'bothtune':                     
                    control_code = None
                    print('433::config.optim_prefix, optim_prefix_bool=', config.optim_prefix, optim_prefix_bool)
                    prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=1)

                    prompt = [x.expand(-1, args.num_return_sequences , -1, -1, -1) for x in prompt]
                    print('437::prompt[0].shape=', prompt[0].shape)
                    print('\n\n401::input_ids.shape=', input_ids.shape, ', input_ids=', input_ids, '\n\n')
                    print('439::type(prompt)=', type(prompt), ', len(prompt)=', len(prompt))


                    if decode_mode == 'beam':
                        print("443::decode_mode == 'beam'")
                        output_sequences = gpt2.generate(
                            input_ids=input_ids,
                            emb_match=None,
                            control_code=None,
                            past_key_values=prompt,
                            max_length=args.length + len(encoded_prompt[0]),
                            min_length=2, #min_length=5,
                            temperature=args.temperature,
                            top_k=args.k,
                            top_p=0.9, #top_p=0.5,
                            repetition_penalty=args.repetition_penalty, ##args.repetition_penalty,
                            do_sample=False,
                            num_beams=5,
                            bad_words_ids=[[628], [198]] if True else None,
                            num_return_sequences=1,
                        )
                        print('460::output_sequences=', output_sequences)


                # Remove the batch dimension when returning multiple sequences
                if len(output_sequences.shape) > 2:
                    output_sequences.squeeze_()
                generated_sequences = []

                for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                    print("469::=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                    # args.stop_token = tokenizer.eos_token
                    generated_sequence = generated_sequence.tolist()

                    # Decode text
                    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                    print('475::text=', text)
                    text_output = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                    idx = text_output.find(tokenizer.eos_token)
                    if idx >= 0:
                        text_output = text_output[:idx]
                    text_output = text_output.strip()
                    print('481::text_output=', text_output)

                    if args.task_mode == 'topic' or args.task_mode == 'sentiment':
                        text_output = prompt_text + ' ' + text_output + ' [SPECIAL_END]'
                    if text_output:
                        print(text_output, file=out_handle)
                    else:
                        print('Error', file=out_handle)

                #end of gen

                out_text = text_output

                cur_line = all_lines[i] + translate_bar + out_text
            
            form_2 = form_1
            form_1 = cur_line            



    # for prompt_idx, prompt_text in enumerate(prompt_text_lst):
        
    #     if args.task_mode == 'nct':
    #         # already processed
    #         encoded_prompt = torch.LongTensor(prompt_text).unsqueeze(0)
    #         print('377::encoded_prompt.shape=', encoded_prompt.shape)

    #     encoded_prompt = encoded_prompt.to(args.device)

    #     if encoded_prompt.size()[-1] == 0:
    #         input_ids = None
    #     else:
    #         input_ids = encoded_prompt

    #     if args.control_mode == 'yes' and args.control_dataless != 'yes':
    #         # URGENT, check whether the next line is necessary?
    #         # control_code = torch.LongTensor(control_codes[prompt_idx]).to(model.device).unsqueeze(0).expand(args.num_return_sequences, -1)
    #         control_code = None
    #         pass
    #     else:
    #         control_code = None

    #     if args.tuning_mode == 'prefixtune' or args.tuning_mode == 'bothtune':                     
    #         control_code = None
    #         print('396::config.optim_prefix, optim_prefix_bool=', config.optim_prefix, optim_prefix_bool)
    #         prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=1)

    #         prompt = [x.expand(-1, args.num_return_sequences , -1, -1, -1) for x in prompt]
    #         print('400::prompt[0].shape=', prompt[0].shape)
    #         print('\n\n401::input_ids.shape=', input_ids.shape, ', input_ids=', input_ids, '\n\n')
    #         print('402::type(prompt)=', type(prompt), ', len(prompt)=', len(prompt))


    #         if decode_mode == 'beam':
    #             print("419::decode_mode == 'beam'")
    #             output_sequences = gpt2.generate(
    #                 input_ids=input_ids,
    #                 emb_match=None,
    #                 control_code=None,
    #                 past_key_values=prompt,
    #                 max_length=args.length + len(encoded_prompt[0]),
    #                 min_length=2, #min_length=5,
    #                 temperature=args.temperature,
    #                 top_k=args.k,
    #                 top_p=0.9, #top_p=0.5,
    #                 repetition_penalty=args.repetition_penalty, ##args.repetition_penalty,
    #                 do_sample=False,
    #                 num_beams=5,
    #                 bad_words_ids=[[628], [198]] if True else None,
    #                 num_return_sequences=1,
    #             )
    #             print('436::output_sequences=', output_sequences)


    #     # Remove the batch dimension when returning multiple sequences
    #     if len(output_sequences.shape) > 2:
    #         output_sequences.squeeze_()
    #     generated_sequences = []

    #     for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    #         print("445::=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
    #         # args.stop_token = tokenizer.eos_token
    #         generated_sequence = generated_sequence.tolist()

    #         # Decode text
    #         text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    #         print('451::text=', text)
    #         text_output = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
    #         idx = text_output.find(tokenizer.eos_token)
    #         if idx >= 0:
    #             text_output = text_output[:idx]
    #         text_output = text_output.strip()
    #         print('457::text_output=', text_output)

    #         if args.task_mode == 'topic' or args.task_mode == 'sentiment':
    #             text_output = prompt_text + ' ' + text_output + ' [SPECIAL_END]'
    #         if text_output:
    #             print(text_output, file=out_handle)
    #         else:
    #             print('Error', file=out_handle)
    # # return generated_sequences


    # if not QUICK_CHECK:
    #     out_handle.close()

if __name__ == "__main__":
    main()
