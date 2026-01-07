# coding=utf-8
# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from typing import Optional, Union

from ...configuration_utils import PreTrainedConfig


class MoonshineStreamingConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MoonshineStreamingModel`]. It is used to
    instantiate a Moonshine Streaming model according to the specified arguments, defining the model architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the decoder.
        encoder_dim (`int`, *optional*, defaults to 288):
            Encoder hidden dimension.
        decoder_dim (`int`, *optional*, defaults to 288):
            Decoder hidden dimension.
        hidden_size (`int`, *optional*, defaults to 288):
            Alias for `decoder_dim` for compatibility with common config helpers.
        head_dim (`int`, *optional*, defaults to 36):
            Dimension per attention head.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads shared by encoder and decoder if per-block values are not set.
        encoder_num_attention_heads (`int`, *optional*):
            Number of attention heads in encoder blocks. Defaults to `num_attention_heads`.
        decoder_num_attention_heads (`int`, *optional*):
            Number of attention heads in decoder blocks. Defaults to `num_attention_heads`.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Alias for `decoder_num_hidden_layers` for compatibility with common config helpers.
        encoder_num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of decoder layer pairs.
        ffn_mult (`int`, *optional*, defaults to 4):
            Multiplier for the inner feed-forward dimension (inner_dim = dim * ffn_mult).
        use_swiglu_encoder (`bool`, *optional*, defaults to `False`):
            Whether to use SwiGLU activations in the encoder feed-forward blocks.
        use_swiglu_decoder (`bool`, *optional*, defaults to `True`):
            Whether to use SwiGLU activations in the decoder feed-forward blocks.
        encoder_rotary_dim (`int`, *optional*, defaults to 0):
            Rotary embedding dimension for encoder self-attention (0 disables rotary).
        decoder_rotary_dim (`int`, *optional*, defaults to 32):
            Rotary embedding dimension for decoder self-attention.
        rotary_base (`float`, *optional*, defaults to 10000.0):
            Base for rotary embeddings.
        rotary_interpolation_factor (`float`, *optional*, defaults to 1.0):
            Interpolation factor for rotary embeddings.
        encoder_window (`tuple[int, int]` or `list[tuple[int, int]]`, *optional*, defaults to `(16, 4)`):
            Sliding window attention specification for encoder layers. Use `None` for full attention.
        attn_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        ff_dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability for feed-forward blocks.
        adapter_dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability for adapter block dropout.
        adapter_max_positions (`int`, *optional*, defaults to 4096):
            Maximum sequence length for adapter positional embeddings.
        adapter_block_size (`int`, *optional*, defaults to 4):
            Block size (in frames) for structured adapter dropout.
        sample_rate (`int`, *optional*, defaults to 16000):
            Audio sample rate expected by the preprocessor.
        frame_ms (`float`, *optional*, defaults to 5.0):
            Frame length in milliseconds for the preprocessor.
        preprocessor_c1 (`int`, *optional*):
            Channel count of the first causal conv layer. Defaults to `2 * encoder_dim`.
        preprocessor_c2 (`int`, *optional*):
            Channel count of the second causal conv layer. Defaults to `encoder_dim`.
        preprocessor_k1 (`int`, *optional*, defaults to 5):
            Kernel size of the first causal conv layer.
        preprocessor_k2 (`int`, *optional*, defaults to 5):
            Kernel size of the second causal conv layer.
        preprocessor_input_dropout_p (`float`, *optional*, defaults to 0.1):
            Probability of replacing input frames with noise during training.
        preprocessor_input_dropout_sigma (`float`, *optional*, defaults to 0.3):
            Standard deviation of Gaussian noise for input dropout.
        preprocessor_asinh_k_init (`float`, *optional*, defaults to 0.75):
            Initial scale for asinh compression.
        num_tokens_per_sec (`float`, *optional*, defaults to 6.5):
            Expected tokens per second for length estimation in greedy generation.
        attn_backend (`str`, *optional*, defaults to `"auto"`):
            Attention backend to use ("auto", "flash", "efficient", or "math").
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether attention projection layers use bias terms.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return and use KV cache.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning-of-sequence token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End-of-sequence token id.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        decoder_start_token_id (`int`, *optional*):
            Decoder start token id. Defaults to `bos_token_id`.
    """

    model_type = "moonshine_streaming"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "decoder_dim",
        "num_attention_heads": "decoder_num_attention_heads",
        "num_hidden_layers": "decoder_num_hidden_layers",
        "attention_dropout": "attn_dropout",
    }

    def __init__(
        self,
        vocab_size: int = 32768,
        encoder_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        hidden_size: int = 288,
        head_dim: int = 36,
        num_attention_heads: int = 8,
        encoder_num_attention_heads: Optional[int] = None,
        decoder_num_attention_heads: Optional[int] = None,
        num_hidden_layers: int = 6,
        encoder_num_hidden_layers: Optional[int] = None,
        decoder_num_hidden_layers: Optional[int] = None,
        ffn_mult: int = 4,
        use_swiglu_encoder: bool = False,
        use_swiglu_decoder: bool = True,
        encoder_rotary_dim: int = 0,
        decoder_rotary_dim: int = 32,
        rotary_base: float = 10000.0,
        rotary_interpolation_factor: float = 1.0,
        encoder_window: Optional[Union[tuple[int, int], list[tuple[int, int]]]] = (16, 4),
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        adapter_dropout: float = 0.1,
        adapter_max_positions: int = 4096,
        adapter_block_size: int = 4,
        sample_rate: int = 16000,
        frame_ms: float = 5.0,
        preprocessor_c1: Optional[int] = None,
        preprocessor_c2: Optional[int] = None,
        preprocessor_k1: int = 5,
        preprocessor_k2: int = 5,
        preprocessor_input_dropout_p: float = 0.1,
        preprocessor_input_dropout_sigma: float = 0.3,
        preprocessor_asinh_k_init: float = 0.75,
        num_tokens_per_sec: float = 6.5,
        attn_backend: str = "auto",
        attention_bias: bool = False,
        pad_head_dim_to_multiple_of: Optional[int] = None,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        decoder_start_token_id: Optional[int] = None,
        **kwargs,
    ):
        if decoder_dim is None:
            decoder_dim = hidden_size
        if encoder_dim is None:
            encoder_dim = decoder_dim

        if encoder_num_attention_heads is None:
            encoder_num_attention_heads = num_attention_heads
        if decoder_num_attention_heads is None:
            decoder_num_attention_heads = num_attention_heads

        if encoder_num_hidden_layers is None:
            encoder_num_hidden_layers = num_hidden_layers
        if decoder_num_hidden_layers is None:
            decoder_num_hidden_layers = num_hidden_layers

        if preprocessor_c1 is None:
            preprocessor_c1 = 2 * encoder_dim
        if preprocessor_c2 is None:
            preprocessor_c2 = encoder_dim

        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.head_dim = head_dim
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.ffn_mult = ffn_mult
        self.use_swiglu_encoder = use_swiglu_encoder
        self.use_swiglu_decoder = use_swiglu_decoder
        self.encoder_rotary_dim = encoder_rotary_dim
        self.decoder_rotary_dim = decoder_rotary_dim
        self.rotary_base = rotary_base
        self.rotary_interpolation_factor = rotary_interpolation_factor
        if isinstance(encoder_window, list):
            if len(encoder_window) == 2 and all(isinstance(value, int) for value in encoder_window):
                encoder_window = tuple(encoder_window)
            else:
                encoder_window = [tuple(window) if isinstance(window, list) else window for window in encoder_window]
        self.encoder_window = encoder_window
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.adapter_dropout = adapter_dropout
        self.adapter_max_positions = adapter_max_positions
        self.adapter_block_size = adapter_block_size
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.preprocessor_c1 = preprocessor_c1
        self.preprocessor_c2 = preprocessor_c2
        self.preprocessor_k1 = preprocessor_k1
        self.preprocessor_k2 = preprocessor_k2
        self.preprocessor_input_dropout_p = preprocessor_input_dropout_p
        self.preprocessor_input_dropout_sigma = preprocessor_input_dropout_sigma
        self.preprocessor_asinh_k_init = preprocessor_asinh_k_init
        self.num_tokens_per_sec = num_tokens_per_sec
        self.attn_backend = attn_backend
        self.attention_bias = attention_bias
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of
        self.use_cache = use_cache

        kwargs.setdefault("is_encoder_decoder", True)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )


__all__ = ["MoonshineStreamingConfig"]
