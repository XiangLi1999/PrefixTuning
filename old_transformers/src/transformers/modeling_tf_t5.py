# coding=utf-8
# Copyright 2018 T5 Authors and The HuggingFace Inc. team.
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
""" TF 2.0 T5 model. """


import copy
import itertools
import math
import warnings

import tensorflow as tf

from .configuration_t5 import T5Config
from .file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_tf_outputs import TFSeq2SeqLMOutput, TFSeq2SeqModelOutput
from .modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    cast_bool_to_primitive,
    keras_serializable,
    shape_list,
)
from .tokenization_utils import BatchEncoding
from .utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"

TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]

####################################################
# TF 2.0 Models are constructed using Keras imperative API by sub-classing
# - tf.keras.layers.Layer for the layers and
# - TFPreTrainedModel for the models (it-self a sub-class of tf.keras.Model)
####################################################


class TFT5LayerNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        """Construct a layernorm module in the T5 style
        No bias and no substraction of mean.
        """
        super().__init__(**kwargs)
        self.variance_epsilon = epsilon

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.weight = self.add_weight("weight", shape=(input_shape[-1],), initializer="ones")
        super().build(input_shape)

    def call(self, x):
        variance = tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class TFT5DenseReluDense(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.wi = tf.keras.layers.Dense(config.d_ff, use_bias=False, name="wi")
        self.wo = tf.keras.layers.Dense(config.d_model, use_bias=False, name="wo")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.act = tf.keras.activations.relu

    def call(self, hidden_states, training=False):
        h = self.wi(hidden_states)
        h = self.act(h)
        h = self.dropout(h, training=training)
        h = self.wo(h)
        return h


class TFT5LayerFF(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.DenseReluDense = TFT5DenseReluDense(config, name="DenseReluDense")
        self.layer_norm = TFT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, hidden_states, training=False):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x, training=training)
        layer_output = hidden_states + self.dropout(y, training=training)
        return layer_output


class TFT5Attention(tf.keras.layers.Layer):
    NEW_ID = itertools.count()

    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = next(TFT5Attention.NEW_ID)
        self.is_decoder = config.is_decoder
        self.use_cache = config.use_cache
        self.has_relative_attention_bias = has_relative_attention_bias
        self.output_attentions = config.output_attentions

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = tf.keras.layers.Dense(self.inner_dim, use_bias=False, name="q")
        self.k = tf.keras.layers.Dense(self.inner_dim, use_bias=False, name="k")
        self.v = tf.keras.layers.Dense(self.inner_dim, use_bias=False, name="v")
        self.o = tf.keras.layers.Dense(self.d_model, use_bias=False, name="o")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = tf.keras.layers.Embedding(
                self.relative_attention_num_buckets,
                self.n_heads,
                name="relative_attention_bias",
            )
        self.pruned_heads = set()

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += tf.dtypes.cast(tf.math.less(n, 0), tf.int32) * num_buckets
            n = tf.math.abs(n)
        else:
            n = tf.math.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(n, max_exact)
        val_if_large = max_exact + tf.dtypes.cast(
            tf.math.log(tf.dtypes.cast(n, tf.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            tf.int32,
        )
        val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
        ret += tf.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = tf.range(qlen)[:, None]
        memory_position = tf.range(klen)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=not self.is_decoder,
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)  # shape (1, num_heads, qlen, klen)
        return values

    def call(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        cache=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        training=False,
        output_attentions=False,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value_state[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = shape_list(input)

        if past_key_value_state is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert (
                len(past_key_value_state) == 2
            ), "past_key_value_state should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value_state)
            )
            real_qlen = qlen + shape_list(past_key_value_state[0])[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = shape_list(kv)[1]

        def shape(x):
            """  projection """
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, self.d_kv)), perm=(0, 2, 1, 3))

        def unshape(x):
            """  compute context """
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.inner_dim))

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_value_state is None:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_value_state is not None:
            if kv is None:
                k_, v_ = past_key_value_state
                k = tf.concat([k_, k], axis=2)  # (bs, n_heads, klen, dim_per_head)
                v = tf.concat([v_, v], axis=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value_state

        # to cope with keras serialization
        if self.is_decoder and cast_bool_to_primitive(use_cache, self.use_cache) is True:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        scores = tf.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(real_qlen, klen)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value_state is not None:
                position_bias = position_bias[:, :, -1:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = tf.nn.softmax(scores, axis=-1)  # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights, training=training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = tf.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,) + present_key_value_state

        if output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


class TFT5LayerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.SelfAttention = TFT5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            name="SelfAttention",
        )
        self.layer_norm = TFT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y, training=training)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class TFT5LayerCrossAttention(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.EncDecAttention = TFT5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            name="EncDecAttention",
        )
        self.layer_norm = TFT5LayerNorm(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(
        self,
        hidden_states,
        kv,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x,
            mask=attention_mask,
            kv=kv,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=past_key_value_state,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y, training=training)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class TFT5Block(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.is_decoder = config.is_decoder
        self.layer = []
        self.layer.append(
            TFT5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                name="layer_._0",
            )
        )
        if self.is_decoder:
            self.layer.append(
                TFT5LayerCrossAttention(
                    config,
                    has_relative_attention_bias=has_relative_attention_bias,
                    name="layer_._1",
                )
            )

        self.layer.append(TFT5LayerFF(config, name="layer_._{}".format(len(self.layer))))

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):

        if past_key_value_state is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_values,
                "2 (past / key) for cross attention" if expected_num_past_key_values == 4 else "",
                len(past_key_value_state),
            )
            assert len(past_key_value_state) == expected_num_past_key_values, error_message

            self_attn_past_key_value_state = past_key_value_state[:2]
            cross_attn_past_key_value_state = past_key_value_state[2:]
        else:
            self_attn_past_key_value_state, cross_attn_past_key_value_state = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=self_attn_past_key_value_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = shape_list(present_key_value_state[0])[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value_state=cross_attn_past_key_value_state,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, training=training)
        outputs = (hidden_states,)

        # Add attentions if we output them
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class _NoLayerEmbedTokens:
    """
    this class wraps a the TFSharedEmbeddingTokens layer into a python 'no-keras-layer'
    class to avoid problem with weight restoring. Also it makes sure that the layer is
    called from the correct scope to avoid problem with saving/storing the correct weights
    """

    def __init__(self, layer, abs_scope_name=None):
        self._layer = layer
        self._abs_scope_name = abs_scope_name

    def call(self, inputs, mode="embedding"):
        if self._abs_scope_name is None:
            return self._layer.call(inputs, mode)

        # if an abs scope name is given to the embedding variable, call variable from absolute scope
        with tf.compat.v1.variable_scope(self._abs_scope_name, auxiliary_name_scope=False) as abs_scope_name:
            with tf.name_scope(abs_scope_name.original_name_scope):
                return self._layer.call(inputs, mode)

    def __call__(self, inputs, mode="embedding"):
        if self._abs_scope_name is None:
            return self._layer(inputs, mode)

        # if an abs scope name is given to the embedding variable, call variable from absolute scope
        with tf.compat.v1.variable_scope(self._abs_scope_name, auxiliary_name_scope=False) as abs_scope_name:
            with tf.name_scope(abs_scope_name.original_name_scope):
                return self._layer(inputs, mode)


####################################################
# The full model without a specific pretrained or finetuning head is
# provided as a tf.keras.layers.Layer usually called "TFT5MainLayer"
####################################################
@keras_serializable
class TFT5MainLayer(tf.keras.layers.Layer):
    config_class = T5Config

    def __init__(self, config, embed_tokens=None, **kwargs):
        super().__init__(**kwargs)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.config = config
        self.num_hidden_layers = config.num_layers

        self.block = [
            TFT5Block(
                config,
                has_relative_attention_bias=bool(i == 0),
                name="block_._{}".format(i),
            )
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = TFT5LayerNorm(epsilon=config.layer_norm_epsilon, name="final_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    def call(
        self,
        inputs,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        training=False,
        **kwargs,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            encoder_hidden_states = inputs[2] if len(inputs) > 2 else encoder_hidden_states
            encoder_attention_mask = inputs[3] if len(inputs) > 3 else encoder_attention_mask
            inputs_embeds = inputs[4] if len(inputs) > 4 else inputs_embeds
            head_mask = inputs[5] if len(inputs) > 5 else head_mask
            past_key_values = inputs[6] if len(inputs) > 6 else past_key_values
            use_cache = inputs[7] if len(inputs) > 7 else use_cache
            output_attentions = inputs[8] if len(inputs) > 8 else output_attentions
            output_hidden_states = inputs[9] if len(inputs) > 9 else output_hidden_states
            assert len(inputs) <= 10, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            encoder_hidden_states = inputs.get("encoder_hidden_states", encoder_hidden_states)
            encoder_attention_mask = inputs.get("encoder_attention_mask", encoder_attention_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            head_mask = inputs.get("head_mask", head_mask)
            past_key_values = inputs.get("past_key_values", past_key_values)
            use_cache = inputs.get("use_cache", use_cache)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            assert len(inputs) <= 10, "Too many inputs."

            if "past_key_value_states" in inputs:
                warnings.warn(
                    "The `past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                    FutureWarning,
                )
                past_key_values = inputs.pop("past_key_value_states")
        else:
            input_ids = inputs
            if "past_key_value_states" in kwargs:
                warnings.warn(
                    "The `past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                    FutureWarning,
                )
                past_key_values = kwargs.pop("past_key_value_states")

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both inputs and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either inputs or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if past_key_values is not None:
            assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
                input_shape, (batch_size, 1)
            )
            # required mask seq length can be calculated via length of past
            # key value states and seq_length = 1 for the last token
            mask_seq_length = shape_list(past_key_values[0][0])[2] + seq_length
        else:
            mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = tf.fill((batch_size, mask_seq_length), 1)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = shape_list(encoder_hidden_states)[1]
            encoder_attention_mask = tf.fill((batch_size, encoder_seq_length), 1)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        num_dims_attention_mask = len(shape_list(attention_mask))
        if num_dims_attention_mask == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif num_dims_attention_mask == 2:
            # Provided a padding mask of dimensions [batch_size, mask_seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            if self.is_decoder:
                seq_ids = tf.range(mask_seq_length)
                causal_mask = tf.less_equal(
                    tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                    seq_ids[None, :, None],
                )
                causal_mask = tf.cast(causal_mask, dtype=tf.float32)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                if past_key_values[0] is not None:
                    extended_attention_mask = extended_attention_mask[:, :, -1:, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
        # extended_attention_mask = tf.math.equal(extended_attention_mask,
        #                                         tf.transpose(extended_attention_mask, perm=(-1, -2)))

        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        if self.is_decoder and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastabe to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=tf.float32)
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposistion
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            encoder_extended_attention_mask = None

        assert head_mask is None, "Head mask not supported"
        head_mask = [None] * self.num_hidden_layers

        present_key_value_states = ()
        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds, training=training)

        for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
                output_attentions=output_attentions,
                training=training,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # need to check if is decoder here as well for special cases when using keras compile
        if cast_bool_to_primitive(use_cache, self.use_cache) is True and self.is_decoder:
            outputs = outputs + (present_key_value_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


####################################################
# TFT5PreTrainedModel is a sub-class of tf.keras.Model
# which take care of loading and saving pretrained weights
# and various common utilities.
# Here you just need to specify a few (self-explanatory)
# pointers for your model.
####################################################
class TFT5PreTrainedModel(TFPreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = T5Config
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        inputs = tf.constant(DUMMY_INPUTS)
        input_mask = tf.constant(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": inputs,
            "decoder_input_ids": inputs,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In TF T5 it is usually set to the pad_token_id. See T5 docs for more information"

        shifted_input_ids = tf.cast(input_ids, tf.int32)
        shifted_input_ids = tf.roll(shifted_input_ids, 1, axis=-1)
        start_tokens = tf.fill((shape_list(shifted_input_ids)[0], 1), decoder_start_token_id)
        shifted_input_ids = tf.concat([start_tokens, shifted_input_ids[:, 1:]], -1)

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = tf.where(
            shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids
        )

        # "Verify that `labels` has only positive values and -100"
        assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.cast(0, tf.int32))

        # Make sure the assertion op is called by wrapping the result in an identity no-op
        with tf.control_dependencies([assert_gte0]):
            shifted_input_ids = tf.identity(shifted_input_ids)

        return shifted_input_ids


T5_START_DOCSTRING = r"""

    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.
    It's an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

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
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        inputs (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            T5 is a model with relative position embeddings so you should be able to pad the inputs on
            the right or the left.

            Indices can be obtained using :class:`~transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.__call__` and
            :func:`transformers.PreTrainedTokenizer.encode` for details.

            To know more on how to prepare :obj:`inputs` for pre-training take a look at
            `T5 Training <./t5.html#training>`__.
        decoder_input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for sequence to sequence training. T5 uses the :obj:`pad_token_id` as the starting token for
            :obj:`decoder_input_ids` generation.
            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            To know more on how to prepare :obj:`decoder_input_ids` for pretraining take a look at
            `T5 Training <./t5.html#training>`__. If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both
            unset, :obj:`decoder_input_ids` takes the value of :obj:`input_ids`.
        attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        encoder_outputs (:obj:`tuple(tuple(tf.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, :obj:`optional`: `hidden_states`, :obj:`optional`: `attentions`)
            :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)` is a sequence of
            hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        decoder_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        past_key_values (:obj:`tuple(tuple(tf.Tensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            ontains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If set to :obj:`True`, ``past_key_values`` key value states are returned and can be used to speed up
            decoding (see ``past_key_values``).
        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation.
            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds` have to be input
            (see :obj:`past_key_values`).
            This is useful if you want more control over how to convert :obj:`decoder_input_ids` indices into
            associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both
            unset, :obj:`decoder_input_embeds` takes the value of :obj:`input_embeds`.
        head_mask: (:obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

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
    "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
)
class TFT5Model(TFT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model, name="shared")

        # retrieve correct absolute scope for embed token wrapper
        with tf.compat.v1.variable_scope("shared") as shared_abs_scope_name:
            pass

        embed_tokens = _NoLayerEmbedTokens(self.shared, abs_scope_name=shared_abs_scope_name)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = TFT5MainLayer(encoder_config, embed_tokens, name="encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = TFT5MainLayer(decoder_config, embed_tokens, name="decoder")

    def get_input_embeddings(self):
        return self.shared

    def get_output_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared.weight = new_embeddings
        self.shared.vocab_size = self.shared.weight.shape[0]
        # retrieve correct absolute scope for embed token wrapper
        with tf.compat.v1.variable_scope("shared") as shared_abs_scope_name:
            pass
        embed_tokens = _NoLayerEmbedTokens(self.shared, abs_scope_name=shared_abs_scope_name)
        self.encoder.set_embed_tokens(embed_tokens)
        self.decoder.set_embed_tokens(embed_tokens)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        inputs,
        attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, TFT5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = TFT5Model.from_pretrained('t5-small')
            >>> inputs = tokenizer.encode("Hello, my dog is cute", return_tensors="tf")  # Batch size 1
            >>> outputs = model(inputs, decoder_input_ids=inputs)
            >>> last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            encoder_outputs = inputs[2] if len(inputs) > 2 else encoder_outputs
            inputs_embeds = inputs[3] if len(inputs) > 3 else inputs_embeds
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            past_key_values = inputs[5] if len(inputs) > 5 else past_key_values
            decoder_input_ids = inputs[6] if len(inputs) > 6 else decoder_input_ids
            decoder_attention_mask = inputs[7] if len(inputs) > 7 else decoder_attention_mask
            decoder_inputs_embeds = inputs[8] if len(inputs) > 8 else decoder_inputs_embeds
            use_cache = inputs[9] if len(inputs) > 9 else use_cache
            output_attentions = inputs[10] if len(inputs) > 10 else output_attentions
            output_hidden_states = inputs[11] if len(inputs) > 11 else output_hidden_states
            return_dict = inputs[12] if len(inputs) > 12 else return_dict
            assert len(inputs) <= 13, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            if "inputs" in inputs:
                warnings.warn("Using `inputs` as a keyword argument is deprecated. Please use `input_ids` instead.")
                input_ids = inputs.get("inputs")
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            encoder_outputs = inputs.get("encoder_outputs", encoder_outputs)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            head_mask = inputs.get("head_mask", head_mask)
            past_key_values = inputs.get("past_key_values", past_key_values)
            decoder_input_ids = inputs.get("decoder_input_ids", decoder_input_ids)
            decoder_attention_mask = inputs.get("decoder_attention_mask", decoder_attention_mask)
            decoder_inputs_embeds = inputs.get("decoder_inputs_embeds", decoder_inputs_embeds)
            use_cache = inputs.get("use_cache", use_cache)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            assert len(inputs) <= 13, "Too many inputs."

            if "past_key_value_states" in inputs:
                warnings.warn(
                    "The `past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                    FutureWarning,
                )
                past_key_values = inputs.pop("past_key_value_states")
        else:
            input_ids = inputs

            if "past_key_value_states" in kwargs:
                warnings.warn(
                    "The `past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                    FutureWarning,
                )
                past_key_values = kwargs.pop("past_key_value_states")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                [
                    input_ids,
                    attention_mask,
                    None,
                    None,
                    inputs_embeds,
                    head_mask,
                    None,
                    False,
                    output_attentions,
                    output_hidden_states,
                ],
                training=training,
            )

        hidden_states = encoder_outputs[0]

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            [
                decoder_input_ids,
                decoder_attention_mask,
                hidden_states,
                attention_mask,
                decoder_inputs_embeds,
                head_mask,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
            ],
            training=training,
        )
        past = (
            (encoder_outputs, decoder_outputs[1]) if cast_bool_to_primitive(use_cache, self.config.use_cache) else None
        )
        if not return_dict:
            if past is not None:
                decoder_outputs = decoder_outputs[:1] + (past,) + decoder_outputs[2:]
            return decoder_outputs + encoder_outputs

        # If put before, this breaks the tf compilation.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # This is long and annoying but if we introduce return_dict at the TFT5MainLayer level (like in PyTorch)
        # TF refuses to compile anymore.
        if not cast_bool_to_primitive(use_cache, self.config.use_cache):
            decoder_outputs = decoder_outputs[:1] + (None,) + decoder_outputs[1:]
        if not cast_bool_to_primitive(output_hidden_states, self.config.output_hidden_states):
            encoder_outputs = encoder_outputs[:1] + (None,) + encoder_outputs[1:]
            decoder_outputs = decoder_outputs[:2] + (None,) + decoder_outputs[2:]
        if not cast_bool_to_primitive(output_attentions, self.config.output_attentions):
            encoder_outputs = encoder_outputs + (None,)
            decoder_outputs = decoder_outputs + (None,)

        return TFSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs[0],
            past_key_values=past,
            decoder_hidden_states=decoder_outputs[2],
            decoder_attentions=decoder_outputs[3],
            encoder_last_hidden_state=encoder_outputs[0],
            encoder_hidden_states=encoder_outputs[1],
            encoder_attentions=encoder_outputs[2],
        )


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class TFT5ForConditionalGeneration(TFT5PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model_dim = config.d_model

        self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model, name="shared")

        # retrieve correct absolute scope for embed token wrapper
        with tf.compat.v1.variable_scope("shared") as shared_abs_scope_name:
            pass

        embed_tokens = _NoLayerEmbedTokens(self.shared, abs_scope_name=shared_abs_scope_name)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = TFT5MainLayer(encoder_config, embed_tokens, name="encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = TFT5MainLayer(decoder_config, embed_tokens, name="decoder")

    def get_input_embeddings(self):
        return self.shared

    def get_output_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared.weight = new_embeddings
        # retrieve correct absolute scope for embed token wrapper
        with tf.compat.v1.variable_scope("shared") as shared_abs_scope_name:
            pass
        embed_tokens = _NoLayerEmbedTokens(self.shared, abs_scope_name=shared_abs_scope_name)
        self.encoder.set_embed_tokens(embed_tokens)
        self.decoder.set_embed_tokens(embed_tokens)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        inputs,
        attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, TFT5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
            >>> inputs = tokenizer.encode("Hello, my dog is cute", return_tensors="tf")  # Batch size 1
            >>> outputs = model(inputs, decoder_input_ids=inputs)
            >>> prediction_scores = outputs[0]

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
            >>> inputs = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="tf")  # Batch size 1
            >>> result = model.generate(inputs)

        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            encoder_outputs = inputs[2] if len(inputs) > 2 else encoder_outputs
            inputs_embeds = inputs[3] if len(inputs) > 3 else inputs_embeds
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            past_key_values = inputs[5] if len(inputs) > 5 else past_key_values
            decoder_input_ids = inputs[6] if len(inputs) > 6 else decoder_input_ids
            decoder_attention_mask = inputs[7] if len(inputs) > 7 else decoder_attention_mask
            decoder_inputs_embeds = inputs[8] if len(inputs) > 8 else decoder_inputs_embeds
            use_cache = inputs[9] if len(inputs) > 9 else use_cache
            output_attentions = inputs[10] if len(inputs) > 10 else output_attentions
            output_hidden_states = inputs[11] if len(inputs) > 11 else output_hidden_states
            return_dict = inputs[12] if len(inputs) > 12 else return_dict
            labels = inputs[13] if len(inputs) > 13 else labels
            assert len(inputs) <= 14, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            if "inputs" in inputs:
                warnings.warn("Using `inputs` as a keyword argument is deprecated. Please use `input_ids` instead.")
                input_ids = inputs.get("inputs")
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            encoder_outputs = inputs.get("encoder_outputs", encoder_outputs)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            head_mask = inputs.get("head_mask", head_mask)
            past_key_values = inputs.get("past_key_values", past_key_values)
            decoder_input_ids = inputs.get("decoder_input_ids", decoder_input_ids)
            decoder_attention_mask = inputs.get("decoder_attention_mask", decoder_attention_mask)
            decoder_inputs_embeds = inputs.get("decoder_inputs_embeds", decoder_inputs_embeds)
            use_cache = inputs.get("use_cache", use_cache)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            return_dict = inputs.get("return_dict", return_dict)
            labels = inputs.get("labels", labels)
            assert len(inputs) <= 14, "Too many inputs."

            if "past_key_value_states" in inputs:
                warnings.warn(
                    "The `past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                    FutureWarning,
                )
                past_key_values = inputs.pop("past_key_value_states")
        else:
            input_ids = inputs

            if "past_key_value_states" in kwargs:
                warnings.warn(
                    "The `past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                    FutureWarning,
                )
                past_key_values = kwargs.pop("past_key_value_states")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                [
                    input_ids,
                    attention_mask,
                    None,
                    None,
                    inputs_embeds,
                    head_mask,
                    None,
                    False,
                    output_attentions,
                    output_hidden_states,
                ],
                training=training,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            [
                decoder_input_ids,
                decoder_attention_mask,
                hidden_states,
                attention_mask,
                decoder_inputs_embeds,
                head_mask,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
            ],
            training=training,
        )

        sequence_output = decoder_outputs[0] * (self.model_dim ** -0.5)
        embed_tokens = self.get_output_embeddings()
        logits = embed_tokens(sequence_output, mode="linear")

        loss = None if labels is None else self.compute_loss(labels, logits)

        past = (
            (encoder_outputs, decoder_outputs[1]) if cast_bool_to_primitive(use_cache, self.config.use_cache) else None
        )
        if not return_dict:
            if past is not None:
                decoder_outputs = decoder_outputs[:1] + (past,) + decoder_outputs[2:]
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        # Putting this before breaks tf compilation.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # This is long and annoying but if we introduce return_dict at the TFT5MainLayer level (like in PyTorch)
        # TF refuses to compile anymore.
        if not cast_bool_to_primitive(use_cache, self.config.use_cache):
            decoder_outputs = decoder_outputs[:1] + (None,) + decoder_outputs[1:]
        if not cast_bool_to_primitive(output_hidden_states, self.config.output_hidden_states):
            encoder_outputs = encoder_outputs[:1] + (None,) + encoder_outputs[1:]
            decoder_outputs = decoder_outputs[:2] + (None,) + decoder_outputs[2:]
        if not cast_bool_to_primitive(output_attentions, self.config.output_attentions):
            encoder_outputs = encoder_outputs + (None,)
            decoder_outputs = decoder_outputs + (None,)

        return TFSeq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past,
            decoder_hidden_states=decoder_outputs[2],
            decoder_attentions=decoder_outputs[3],
            encoder_last_hidden_state=encoder_outputs[0],
            encoder_hidden_states=encoder_outputs[1],
            encoder_attentions=encoder_outputs[2],
        )

    def prepare_inputs_for_generation(self, inputs, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if len(past) < 2:
            encoder_outputs, past_key_values = past, None
        else:
            encoder_outputs, past_key_values = past[0], past[1]

        return {
            "inputs": None,  # inputs don't have to be defined, but still need to be passed to make Keras.layer.__call__ happy
            "decoder_input_ids": inputs,  # inputs are the decoder_input_ids
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder

        if len(past) < 2:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        decoder_past = past[1]
        past = (past[0],)
        reordered_decoder_past = ()

        for layer_past_states in decoder_past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (tf.gather(layer_past_state, beam_idx),)

            assert shape_list(reordered_layer_past_states[0]) == shape_list(layer_past_states[0])
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return past + (reordered_decoder_past,)
