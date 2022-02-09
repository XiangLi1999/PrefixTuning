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
""" TF 2.0 XLNet model.
"""


from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf

from .activations_tf import get_tf_activation
from .configuration_xlnet import XLNetConfig
from .file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFSharedEmbeddings,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    shape_list,
)
from .tokenization_utils import BatchEncoding
from .utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "XLNetConfig"
_TOKENIZER_FOR_DOC = "XLNetTokenizer"

TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlnet-base-cased",
    "xlnet-large-cased",
    # See all XLNet models at https://huggingface.co/models?filter=xlnet
]


class TFXLNetRelativeAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if config.d_model % config.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.d_model, config.n_head)
            )

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def build(self, input_shape):
        initializer = get_initializer(self.initializer_range)
        self.q = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="q"
        )
        self.k = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="k"
        )
        self.v = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="v"
        )
        self.o = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="o"
        )
        self.r = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="r"
        )
        self.r_r_bias = self.add_weight(
            shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
        )
        self.r_s_bias = self.add_weight(
            shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_s_bias"
        )
        self.r_w_bias = self.add_weight(
            shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
        )
        self.seg_embed = self.add_weight(
            shape=(2, self.n_head, self.d_head), initializer=initializer, trainable=True, name="seg_embed"
        )
        super().build(input_shape)

    def prune_heads(self, heads):
        raise NotImplementedError

    def rel_shift(self, x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = shape_list(x)

        x = tf.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
        x = x[1:, ...]
        x = tf.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
        x = x[:, 0:klen, :, :]
        # x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))

        return x

    def rel_attn_core(
        self, q_head, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask, head_mask, output_attentions, training=False
    ):
        """Core relative positional attention operations."""
        # content based attention score
        ac = tf.einsum("ibnd,jbnd->ijbn", q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = tf.einsum("ibnd,jbnd->ijbn", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=shape_list(ac)[1])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = tf.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = tf.einsum("ijbs,ibns->ijbn", seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            if attn_mask.dtype == tf.float16:
                attn_score = attn_score - 65500 * attn_mask
            else:
                attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = tf.nn.softmax(attn_score, axis=1)

        attn_prob = self.dropout(attn_prob, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # attention output
        attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, v_head_h)

        if output_attentions:
            return attn_vec, attn_prob

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True, training=False):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        attn_out = tf.einsum("ibnd,hnd->ibh", attn_vec, self.o)

        attn_out = self.dropout(attn_out, training=training)

        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def call(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems,
        target_mapping,
        head_mask,
        output_attentions,
        training=False,
    ):
        if g is not None:
            # Two-stream attention with relative positional encoding.
            # content based attention score
            if mems is not None and len(shape_list(mems)) > 1:
                cat = tf.concat([mems, h], axis=0)
            else:
                cat = h

            # content-based key head
            k_head_h = tf.einsum("ibh,hnd->ibnd", cat, self.k)

            # content-based value head
            v_head_h = tf.einsum("ibh,hnd->ibnd", cat, self.v)

            # position-based key head
            k_head_r = tf.einsum("ibh,hnd->ibnd", r, self.r)

            # h-stream
            # content-stream query head
            q_head_h = tf.einsum("ibh,hnd->ibnd", h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat,
                attn_mask_h,
                head_mask,
                output_attentions,
                training=training,
            )

            if output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            # post processing
            output_h = self.post_attention(h, attn_vec_h, training=training)

            # g-stream
            # query-stream query head
            q_head_g = tf.einsum("ibh,hnd->ibnd", g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = tf.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat,
                    attn_mask_g,
                    head_mask,
                    output_attentions,
                    training=training,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = tf.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat,
                    attn_mask_g,
                    head_mask,
                    output_attentions,
                    training=training,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # post processing
            output_g = self.post_attention(g, attn_vec_g, training=training)

            if output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            # Multi-head attention with relative positional encoding
            if mems is not None and len(shape_list(mems)) > 1:
                cat = tf.concat([mems, h], axis=0)
            else:
                cat = h

            # content heads
            q_head_h = tf.einsum("ibh,hnd->ibnd", h, self.q)
            k_head_h = tf.einsum("ibh,hnd->ibnd", cat, self.k)
            v_head_h = tf.einsum("ibh,hnd->ibnd", cat, self.v)

            # positional heads
            k_head_r = tf.einsum("ibh,hnd->ibnd", r, self.r)

            # core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat,
                attn_mask_h,
                head_mask,
                output_attentions,
                training=training,
            )

            if output_attentions:
                attn_vec, attn_prob = attn_vec

            # post processing
            output_h = self.post_attention(h, attn_vec, training=training)
            output_g = None

        outputs = (output_h, output_g)
        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class TFXLNetFeedForward(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.layer_1 = tf.keras.layers.Dense(
            config.d_inner, kernel_initializer=get_initializer(config.initializer_range), name="layer_1"
        )
        self.layer_2 = tf.keras.layers.Dense(
            config.d_model, kernel_initializer=get_initializer(config.initializer_range), name="layer_2"
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = get_tf_activation(config.ff_activation)
        else:
            self.activation_function = config.ff_activation

    def call(self, inp, training=False):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output, training=training)
        output = self.layer_2(output)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output + inp)
        return output


class TFXLNetLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.rel_attn = TFXLNetRelativeAttention(config, name="rel_attn")
        self.ff = TFXLNetFeedForward(config, name="ff")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(
        self,
        output_h,
        output_g,
        non_tgt_mask,
        attn_mask,
        pos_emb,
        seg_mat,
        mems,
        target_mapping,
        head_mask,
        output_attentions,
        training=False,
    ):
        outputs = self.rel_attn(
            output_h,
            output_g,
            non_tgt_mask,
            attn_mask,
            pos_emb,
            seg_mat,
            mems,
            target_mapping,
            head_mask,
            output_attentions,
            training=training,
        )
        output_h, output_g = outputs[:2]

        if output_g is not None:
            output_g = self.ff(output_g, training=training)
        output_h = self.ff(output_h, training=training)

        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if there are there
        return outputs


class TFXLNetLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


@keras_serializable
class TFXLNetMainLayer(tf.keras.layers.Layer):
    config_class = XLNetConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.return_dict

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        self.use_bfloat16 = config.use_bfloat16
        self.initializer_range = config.initializer_range

        self.word_embedding = TFSharedEmbeddings(
            config.vocab_size, config.d_model, initializer_range=config.initializer_range, name="word_embedding"
        )
        self.layer = [TFXLNetLayer(config, name="layer_._{}".format(i)) for i in range(config.n_layer)]
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def get_input_embeddings(self):
        return self.word_embedding

    def set_input_embeddings(self, value):
        self.word_embedding.weight = value
        self.word_embedding.vocab_size = value.shape[0]

    def build(self, input_shape):
        initializer = get_initializer(self.initializer_range)
        self.mask_emb = self.add_weight(
            shape=(1, 1, self.d_model), initializer=initializer, trainable=True, name="mask_emb"
        )

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen, dtype=tf.float32):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: TODO Lysandre didn't fill
            mlen: TODO Lysandre didn't fill

        ::

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]

        """
        attn_mask = tf.ones([qlen, qlen], dtype=dtype)
        mask_u = tf.matrix_band_part(attn_mask, 0, -1)
        mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
        attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
        ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
        if self.same_length:
            mask_l = tf.matrix_band_part(attn_mask, -1, 0)
            ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
        return ret

    def cache_mem(self, curr_out, prev_mem):
        """cache hidden states into memory."""
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        if prev_mem is None:
            new_mem = curr_out[-self.mem_len :]
        else:
            new_mem = tf.concat([prev_mem, curr_out], 0)[-self.mem_len :]

        return tf.stop_gradient(new_mem)

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = tf.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = tf.tile(pos_emb, [1, bsz, 1])

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None, dtype=None):
        """create relative positional encoding."""
        freq_seq = tf.range(0, self.d_model, 2.0)
        if dtype is not None and dtype != tf.float32:
            freq_seq = tf.cast(freq_seq, dtype=dtype)
        inv_freq = 1 / (10000 ** (freq_seq / self.d_model))

        if self.attn_type == "bi":
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError("Unknown `attn_type` {}.".format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            bwd_pos_seq = tf.range(-beg, -end, 1.0)

            if dtype is not None and dtype != tf.float32:
                fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
                bwd_pos_seq = tf.cast(bwd_pos_seq, dtype=dtype)

            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
                bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len, self.clamp_len)

            if bsz is not None:
                assert bsz % 2 == 0, f"With bi_data, the batch size {bsz} should be divisible by 2"
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
        else:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            if dtype is not None and dtype != tf.float32:
                fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    def call(
        self,
        inputs,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            mems = inputs[2] if len(inputs) > 2 else mems
            perm_mask = inputs[3] if len(inputs) > 3 else perm_mask
            target_mapping = inputs[4] if len(inputs) > 4 else target_mapping
            token_type_ids = inputs[5] if len(inputs) > 5 else token_type_ids
            input_mask = inputs[6] if len(inputs) > 6 else input_mask
            head_mask = inputs[7] if len(inputs) > 7 else head_mask
            inputs_embeds = inputs[8] if len(inputs) > 8 else inputs_embeds
            use_cache = inputs[9] if len(inputs) > 9 else use_cache
            output_attentions = inputs[10] if len(inputs) > 10 else output_attentions
            output_hidden_states = inputs[11] if len(inputs) > 11 else output_hidden_states
            return_dict = inputs[12] if len(inputs) > 12 else return_dict
            assert len(inputs) <= 13, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            mems = inputs.get("mems", mems)
            perm_mask = inputs.get("perm_mask", perm_mask)
            target_mapping = inputs.get("target_mapping", target_mapping)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            input_mask = inputs.get("input_mask", input_mask)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            use_cache = inputs.get("use_cache", use_cache)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            assert len(inputs) <= 13, "Too many inputs."
        else:
            input_ids = inputs

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.return_dict

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = tf.transpose(input_ids, perm=(1, 0))
            qlen, bsz = shape_list(input_ids)[:2]
        elif inputs_embeds is not None:
            inputs_embeds = tf.transpose(inputs_embeds, perm=(1, 0, 2))
            qlen, bsz = shape_list(inputs_embeds)[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_type_ids = tf.transpose(token_type_ids, perm=(1, 0)) if token_type_ids is not None else None
        input_mask = tf.transpose(input_mask, perm=(1, 0)) if input_mask is not None else None
        attention_mask = tf.transpose(attention_mask, perm=(1, 0)) if attention_mask is not None else None
        perm_mask = tf.transpose(perm_mask, perm=(1, 2, 0)) if perm_mask is not None else None
        target_mapping = tf.transpose(target_mapping, perm=(1, 2, 0)) if target_mapping is not None else None

        mlen = shape_list(mems[0])[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = tf.bfloat16 if self.use_bfloat16 else tf.float32

        # Attention mask
        # causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError("Unsupported attention type: {}".format(self.attn_type))

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, (
            "You can only use one of input_mask (uses 1 for padding) "
            "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        )
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - tf.cast(attention_mask, dtype=dtype_float)
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = tf.zeros([shape_list(data_mask)[0], mlen, bsz], dtype=dtype_float)
                data_mask = tf.concat([mems_mask, data_mask], axis=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask > 0, dtype=dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -tf.eye(qlen, dtype=dtype_float)
            if mlen > 0:
                non_tgt_mask = tf.concat([tf.zeros([qlen, mlen], dtype=dtype_float), non_tgt_mask], axis=-1)
            non_tgt_mask = tf.cast((attn_mask + non_tgt_mask[:, :, None, None]) > 0, dtype=dtype_float)
        else:
            non_tgt_mask = None

        # Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k, training=training)
        if target_mapping is not None:
            word_emb_q = tf.tile(self.mask_emb, [shape_list(target_mapping)[0], bsz, 1])
            # else:  # We removed the inp_q input which was same as target mapping
            #     inp_q_ext = inp_q[:, :, None]
            #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q, training=training)
        else:
            output_g = None

        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = tf.zeros([mlen, bsz], dtype=tf.int32)
                cat_ids = tf.concat([mem_pad, token_type_ids], 0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = tf.cast(tf.logical_not(tf.equal(token_type_ids[:, None], cat_ids[None, :])), tf.int32)
            seg_mat = tf.one_hot(seg_mat, 2, dtype=dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz, dtype=dtype_float)
        pos_emb = self.dropout(pos_emb, training=training)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            # cache new mems
            if self.mem_len is not None and self.mem_len > 0 and use_cache:
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(
                output_h,
                output_g,
                non_tgt_mask,
                attn_mask,
                pos_emb,
                seg_mat,
                mems[i],
                target_mapping,
                head_mask[i],
                output_attentions,
                training=training,
            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h, training=training)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = tf.transpose(output, perm=(1, 0, 2))

        if not (self.mem_len is not None and self.mem_len > 0 and use_cache):
            new_mems = None
        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(tf.transpose(h, perm=(1, 0, 2)) for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(tf.transpose(hs, perm=(1, 0, 2)) for hs in hidden_states)
        if output_attentions:
            attentions = tuple(tf.transpose(t, perm=(2, 3, 0, 1)) for t in attentions)

        if not return_dict:
            return tuple(v for v in [output, new_mems, hidden_states, attentions] if v is not None)

        return TFXLNetModelOutput(
            last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions
        )


class TFXLNetPreTrainedModel(TFPreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = XLNetConfig
    base_model_prefix = "transformer"


@dataclass
class TFXLNetModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetModel`.

    Args:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: tf.Tensor = None
    mems: Optional[List[tf.Tensor]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFXLNetLMHeadModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetLMHeadModel`.

    Args:
        loss (:obj:`tf.Tensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss (for next-token prediction).
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    mems: Optional[List[tf.Tensor]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFXLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetForSequenceClassification`.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    mems: Optional[List[tf.Tensor]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFXLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetForTokenClassificationOutput`.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    mems: Optional[List[tf.Tensor]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFXLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetForMultipleChoice`.

    Args:
        loss (:obj:`tf.Tensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    mems: Optional[List[tf.Tensor]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFXLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetForQuestionAnsweringSimple`.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    start_logits: tf.Tensor = None
    end_logits: tf.Tensor = None
    mems: Optional[List[tf.Tensor]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


XLNET_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass.
    Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general
    usage and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Parameters:
        config (:class:`~transformers.XLNetConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

XLNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.__call__` and
            :func:`transformers.PreTrainedTokenizer.encode` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (see :obj:`mems` output below) . Can be used to speed up sequential
            decoding. The token ids which have their past given to this model should not be passed as
            :obj:`input_ids` as they have already been computed.

            :obj:`use_cache` has to be set to :obj:`True` to make use of :obj:`mems`.
        perm_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length, sequence_length)`, `optional`):
            Mask to indicate the attention pattern for each input token with values selected in ``[0, 1]``:

            - if ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
            - if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.

            If not set, each token attends to all the others (full bidirectional attention).
            Only used during pretraining (to define factorization order) or for sequential decoding (generation).
        target_mapping (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, num_predict, sequence_length)`, `optional`):
            Mask to indicate the output tokens to use.
            If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        input_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Negative of :obj:`attention_mask`, i.e. with 0 for real tokens and 1 for padding which is kept for
            compatibility with the original code base.

            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **masked**,
            - 0 for tokens that are **not maked**.

            You can only uses one of :obj:`input_mask` and :obj:`attention_mask`.
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare XLNet Model transformer outputing raw hidden-states without any specific head on top.",
    XLNET_START_DOCSTRING,
)
class TFXLNetModel(TFXLNetPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLNetMainLayer(config, name="transformer")

    @add_start_docstrings_to_callable(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlnet-base-cased",
        output_type=TFXLNetModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(self, inputs, **kwargs):
        outputs = self.transformer(inputs, **kwargs)
        return outputs


@add_start_docstrings(
    """XLNet Model with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    XLNET_START_DOCSTRING,
)
class TFXLNetLMHeadModel(TFXLNetPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        self.lm_loss = TFXLNetLMHead(config, self.transformer.word_embedding, name="lm_loss")

    def get_output_embeddings(self):
        return self.lm_loss.input_embeddings

    def prepare_inputs_for_generation(self, inputs, past, **kwargs):
        # Add dummy token at the end (no attention on this one)

        # At every pass, the attention values for the new token and the two last generated tokens
        # are computed, the rest is reloaded from the `past` cache. A purely auto-regressive model would have
        # offset = 1; offset = 2 seems to have slightly better computation.
        offset = 2

        effective_batch_size = inputs.shape[0]
        dummy_token = tf.zeros((effective_batch_size, 1), dtype=tf.int32)

        if past:
            inputs = tf.concat([inputs[:, -offset:], dummy_token], axis=1)
        else:
            inputs = tf.concat([inputs, dummy_token], axis=1)

        # Build permutation mask so that previous tokens don't see last token
        sequence_length = inputs.shape[1]
        perm_mask = tf.zeros((effective_batch_size, sequence_length, sequence_length - 1), dtype=tf.float32)
        perm_mask_seq_end = tf.ones((effective_batch_size, sequence_length, 1), dtype=tf.float32)
        perm_mask = tf.concat([perm_mask, perm_mask_seq_end], axis=-1)

        # We'll only predict the last token
        target_mapping = tf.zeros((effective_batch_size, 1, sequence_length - 1), dtype=tf.float32)
        target_mapping_seq_end = tf.ones((effective_batch_size, 1, 1), dtype=tf.float32)
        target_mapping = tf.concat([target_mapping, target_mapping_seq_end], axis=-1)

        inputs = {
            "inputs": inputs,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "use_cache": kwargs["use_cache"],
        }

        # if past is defined in model kwargs then use it for faster decoding
        if past:
            inputs["mems"] = tuple(layer_past[:-offset, :, :] for layer_past in past)

        return inputs

    @add_start_docstrings_to_callable(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFXLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        inputs,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.

        Return:

        Examples::

            >>> import tensorflow as tf
            >>> import numpy as np
            >>> from transformers import XLNetTokenizer, TFXLNetLMHeadModel

            >>> tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
            >>> model = TFXLNetLMHeadModel.from_pretrained('xlnet-large-cased')

            >>> # We show how to setup inputs to predict a next token using a bi-directional context.
            >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=True))[None, :]  # We will predict the masked token

            >>> perm_mask = np.zeros((1, input_ids.shape[1], input_ids.shape[1]))
            >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token

            >>> target_mapping = np.zeros((1, 1, input_ids.shape[1]))  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

            >>> outputs = model(input_ids, perm_mask=tf.constant(perm_mask, dtype=tf.float32), target_mapping=tf.constant(target_mapping, dtype=tf.float32))

            >>> next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

        """
        return_dict = return_dict if return_dict is not None else self.transformer.return_dict
        if isinstance(inputs, (tuple, list)):
            labels = inputs[13] if len(inputs) > 13 else labels
            if len(inputs) > 13:
                inputs = inputs[:13]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        transformer_outputs = self.transformer(
            inputs,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=True,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
            training=training,
        )
        hidden_state = transformer_outputs[0]
        logits = self.lm_loss(hidden_state, training=training)

        loss = None
        if labels is not None:
            # shift labels to the left and cut last logit token
            logits = logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFXLNetLMHeadModelOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """XLNet Model with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    XLNET_START_DOCSTRING,
)
class TFXLNetForSequenceClassification(TFXLNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.transformer = TFXLNetMainLayer(config, name="transformer")
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
        self.logits_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
        )

    @add_start_docstrings_to_callable(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlnet-base-cased",
        output_type=TFXLNetForSequenceClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.transformer.return_dict
        if isinstance(inputs, (tuple, list)):
            labels = inputs[13] if len(inputs) > 13 else labels
            if len(inputs) > 13:
                inputs = inputs[:13]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        transformer_outputs = self.transformer(
            inputs,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        loss = None if labels is None else self.compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFXLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """XLNET Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    XLNET_START_DOCSTRING,
)
class TFXLNetForMultipleChoice(TFXLNetPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.transformer = TFXLNetMainLayer(config, name="transformer")
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
        self.logits_proj = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
        )

    @property
    def dummy_inputs(self):
        """Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        return {"input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS)}

    @add_start_docstrings_to_callable(XLNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlnet-base-cased",
        output_type=TFXLNetForMultipleChoiceOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs=None,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            mems = inputs[2] if len(inputs) > 2 else mems
            perm_mask = inputs[3] if len(inputs) > 3 else perm_mask
            target_mapping = inputs[4] if len(inputs) > 4 else target_mapping
            token_type_ids = inputs[5] if len(inputs) > 5 else token_type_ids
            input_mask = inputs[6] if len(inputs) > 6 else input_mask
            head_mask = inputs[7] if len(inputs) > 7 else head_mask
            inputs_embeds = inputs[8] if len(inputs) > 8 else inputs_embeds
            use_cache = inputs[9] if len(inputs) > 9 else use_cache
            output_attentions = inputs[10] if len(inputs) > 10 else output_attentions
            output_hidden_states = inputs[11] if len(inputs) > 11 else output_hidden_states
            return_dict = inputs[12] if len(inputs) > 12 else return_dict
            labels = inputs[13] if len(inputs) > 13 else labels
            assert len(inputs) <= 14, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            mems = inputs.get("mems", mems)
            perm_mask = inputs.get("perm_mask", perm_mask)
            target_mapping = inputs.get("target_mapping", target_mapping)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            input_mask = inputs.get("input_mask", input_mask)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            use_cache = inputs.get("use_cache", use_cache)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            labels = inputs.get("labels", labels)
            assert len(inputs) <= 14, "Too many inputs."
        else:
            input_ids = inputs
        return_dict = return_dict if return_dict is not None else self.transformer.return_dict

        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_input_mask = tf.reshape(input_mask, (-1, seq_length)) if input_mask is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        transformer_outputs = self.transformer(
            flat_input_ids,
            flat_attention_mask,
            mems,
            perm_mask,
            target_mapping,
            flat_token_type_ids,
            flat_input_mask,
            head_mask,
            flat_inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.compute_loss(labels, reshaped_logits)

        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFXLNetForMultipleChoiceOutput(
            loss=loss,
            logits=reshaped_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """XLNet Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    XLNET_START_DOCSTRING,
)
class TFXLNetForTokenClassification(TFXLNetPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.transformer = TFXLNetMainLayer(config, name="transformer")
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @add_start_docstrings_to_callable(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlnet-base-cased",
        output_type=TFXLNetForTokenClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.transformer.return_dict
        if isinstance(inputs, (tuple, list)):
            labels = inputs[13] if len(inputs) > 13 else labels
            if len(inputs) > 13:
                inputs = inputs[:13]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        transformer_outputs = self.transformer(
            inputs,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        output = transformer_outputs[0]
        logits = self.classifier(output)
        loss = None if labels is None else self.compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFXLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    XLNET_START_DOCSTRING,
)
class TFXLNetForQuestionAnsweringSimple(TFXLNetPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

    @add_start_docstrings_to_callable(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="xlnet-base-cased",
        output_type=TFXLNetForQuestionAnsweringSimpleOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        inputs=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
        training=False,
    ):
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.transformer.return_dict
        if isinstance(inputs, (tuple, list)):
            start_positions = inputs[13] if len(inputs) > 13 else start_positions
            end_positions = inputs[14] if len(inputs) > 14 else end_positions
            if len(inputs) > 13:
                inputs = inputs[:13]
        elif isinstance(inputs, (dict, BatchEncoding)):
            start_positions = inputs.pop("start_positions", start_positions)
            end_positions = inputs.pop("end_positions", start_positions)

        transformer_outputs = self.transformer(
            inputs,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFXLNetForQuestionAnsweringSimpleOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# @add_start_docstrings("""XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
#     the hidden-states output to compute `span start logits` and `span end logits`). """,
#     XLNET_START_DOCSTRING, XLNET_INPUTS_DOCSTRING)
# class TFXLNetForQuestionAnswering(TFXLNetPreTrainedModel):
#     r"""
#     Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
#         **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``tf.Tensor`` of shape ``(batch_size, config.start_n_top)``
#             Log probabilities for the top config.start_n_top start token possibilities (beam-search).
#         **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``tf.Tensor`` of shape ``(batch_size, config.start_n_top)``
#             Indices for the top config.start_n_top start token possibilities (beam-search).
#         **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``tf.Tensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
#             Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
#         **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``tf.Tensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
#             Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
#         **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
#             ``tf.Tensor`` of shape ``(batch_size,)``
#             Log probabilities for the ``is_impossible`` label of the answers.
#         **mems**:
#             list of ``tf.Tensor`` (one for each layer):
#             that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
#             if config.mem_len > 0 else tuple of None. Can be used to speed up sequential decoding and attend to longer context.
#             See details in the docstring of the `mems` input above.
#         **hidden_states**: (`optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``)
#             list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
#             of shape ``(batch_size, sequence_length, hidden_size)``:
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         **attentions**: (`optional`, returned when ``output_attentions=True``)
#             list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#     Examples::

#         # For example purposes. Not runnable.
#         tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
#         model = XLMForQuestionAnswering.from_pretrained('xlnet-large-cased')
#         input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
#         start_positions = tf.constant([1])
#         end_positions = tf.constant([3])
#         outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
#         loss, start_scores, end_scores = outputs[:2]

#     """
#     def __init__(self, config, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)
#         self.start_n_top = config.start_n_top
#         self.end_n_top = config.end_n_top

#         self.transformer = TFXLNetMainLayer(config, name='transformer')
#         self.start_logits = TFPoolerStartLogits(config, name='start_logits')
#         self.end_logits = TFPoolerEndLogits(config, name='end_logits')
#         self.answer_class = TFPoolerAnswerClass(config, name='answer_class')

#     def call(self, inputs, training=False):
#         transformer_outputs = self.transformer(inputs, training=training)
#         hidden_states = transformer_outputs[0]
#         start_logits = self.start_logits(hidden_states, p_mask=p_mask)

#         outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, let's remove the dimension added by batch splitting
#             for x in (start_positions, end_positions, cls_index, is_impossible):
#                 if x is not None and x.dim() > 1:
#                     x.squeeze_(-1)

#             # during training, compute the end logits based on the ground truth of the start position
#             end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

#             loss_fct = CrossEntropyLoss()
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2

#             if cls_index is not None and is_impossible is not None:
#                 # Predict answerability from the representation of CLS and START
#                 cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
#                 loss_fct_cls = nn.BCEWithLogitsLoss()
#                 cls_loss = loss_fct_cls(cls_logits, is_impossible)

#                 # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
#                 total_loss += cls_loss * 0.5

#             outputs = (total_loss,) + outputs

#         else:
#             # during inference, compute the end logits based on beam search
#             bsz, slen, hsz = hidden_states.size()
#             start_log_probs = F.softmax(start_logits, dim=-1) # shape (bsz, slen)

#             start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1) # shape (bsz, start_n_top)
#             start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz) # shape (bsz, start_n_top, hsz)
#             start_states = torch.gather(hidden_states, -2, start_top_index_exp) # shape (bsz, start_n_top, hsz)
#             start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1) # shape (bsz, slen, start_n_top, hsz)

#             hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states) # shape (bsz, slen, start_n_top, hsz)
#             p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
#             end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
#             end_log_probs = F.softmax(end_logits, dim=1) # shape (bsz, slen, start_n_top)

#             end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1) # shape (bsz, end_n_top, start_n_top)
#             end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
#             end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

#             start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)  # get the representation of START as weighted sum of hidden states
#             cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)  # Shape (batch size,): one single `cls_logits` for each sample

#             outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

#         # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
#         # or (if labels are provided) (total_loss,)
#         return outputs
