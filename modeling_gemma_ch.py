# coding=utf-8
# 版权 2024 Google Inc. 和 HuggingFace Inc. 团队保留所有权利。
#
#
# 根据 Apache 许可证版本 2.0（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 将按“原样”基础分发，不附带任何形式的明示或暗示的保证或条件。
# 请参阅许可证了解特定语言权限和限制。
"""PyTorch Gemma 模型实现。"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN  # 激活函数的映射
from ...cache_utils import Cache, DynamicCache, StaticCache  # 缓存工具
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,  # 注意力掩码转换器
    _prepare_4d_causal_attention_mask,  # 准备4维因果注意力掩码
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast  # 模型输出格式
from ...modeling_utils import PreTrainedModel  # 预训练模型的基类
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13  # PyTorch工具
from ...utils import (
    add_start_docstrings,  # 添加起始文档字符串
    add_start_docstrings_to_model_forward,  # 给模型的forward函数添加起始文档字符串
    is_flash_attn_2_available,  # 检查flash attention 2是否可用
    is_flash_attn_greater_or_equal_2_10,  # 检查flash attention版本是否大于等于2.10
    logging,  # 日志
    replace_return_docstrings,  # 替换返回文档字符串
)
from ...utils.import_utils import is_torch_fx_available  # 检查torch fx是否可用
from .configuration_gemma import GemmaConfig  # Gemma模型配置


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # 引入flash attention功能
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # 引入bert padding处理功能


# 这会将 `_prepare_4d_causal_attention_mask` 函数作为FX图中的叶节点。
# 这意味着该函数不会被追踪，并且仅作为图中的一个节点出现。
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)  # 获取日志记录器

_CONFIG_FOR_DOC = "GemmaConfig"  # 文档中使用的配置名称

# 根据注意力掩码获取未填充数据的函数
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)  # 计算批次中的序列长度
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()  # 获取非零元素的索引
    max_seqlen_in_batch = seqlens_in_batch.max().item()  # 批次中的最大序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))  # 计算累积序列长度
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# RMSNorm层
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # 用于避免除以零的小值
        self.weight = nn.Parameter(torch.zeros(dim))  # 权重参数

    def _norm(self, x):
        # 标准化函数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 前向传播函数
        output = self._norm(x.float()).type_as(x)  # 应用标准化并转换数据类型
        return output * (1 + self.weight)  # 应用缩放

ALL_LAYERNORM_LAYERS.append(GemmaRMSNorm)  # 将RMSNorm层添加到所有LayerNorm层的列表中


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # 初始化旋转位置嵌入模块
        self.dim = dim  # 嵌入维度
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数
        self.base = base  # 基数，用于计算位置频率
        self.register_buffer("inv_freq", None, persistent=False)  # 注册一个不参与参数优化的缓存：逆频率

    @torch.no_grad()  # 前向传播过程中不计算梯度
    def forward(self, x, position_ids, seq_len=None):
        # x: [批次大小, 注意力头数量, 序列长度, 头部大小]
        if self.inv_freq is None:
            # 初始化逆频率，仅在第一次前向传播时计算
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        # 扩展逆频率以匹配位置ID的形状
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # 在长序列上强制使用float32，以避免精度损失
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # 计算旋转嵌入
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        # 返回余弦和正弦嵌入
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """将输入的一半维度旋转。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)  # 将后一半维度旋转后与前一半拼接

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """将旋转位置嵌入应用于查询和键张量。"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # 应用旋转嵌入进行编码
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed  # 返回编码后的查询和键


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 多层感知机（MLP）层
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数

    def forward(self, x):
        # 前向传播，应用门控制和激活函数
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值状态以适配不同数量的注意力头。
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 重复键值状态以匹配注意力头数量
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    """多头注意力机制，来自论文 'Attention Is All You Need'"""

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        # 初始化注意力模块
        self.config = config  # 配置参数
        self.layer_idx = layer_idx  # 层索引，用于缓存
        if layer_idx is None:
            # 如果未传递层索引，给出警告
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
                # f"没有提供`layer_idx`就实例化{self.__class__.__name__}不推荐，如果使用缓存将导致前向调用出错。创建此类时请确保提供`layer_idx`。"
            )

        # 初始化注意力层参数
        self.attention_dropout = config.attention_dropout  # 注意力机制的dropout
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.num_heads = config.num_attention_heads  # 注意力头数量
        self.head_dim = config.head_dim  # 每个注意力头的维度
        self.num_key_value_heads = config.num_key_value_heads  # 键值对的头数量
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 键值对分组的数量
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置嵌入数
        self.rope_theta = config.rope_theta  # 旋转位置嵌入的参数
        self.is_causal = True  # 是否因果关系（即遮掩未来信息）

        # 检查隐藏层大小是否可以被注意力头数量整除
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
                # f"`hidden_size` 必须能被 `num_heads` 整除 (得到 `hidden_size`: {self.hidden_size} 和 `num_heads`: {self.num_heads})."
            )

        # 定义线性层以将输入投影到查询、键和值空间
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )  # 旋转位置嵌入

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # 将隐藏状态投影到查询、键和值空间
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 调整形状以适应多头注意力机制的需要
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 如果提供了旧的键值对，则使用它们来更新状态
        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        if past_key_value is not None:
            # 更新键值对
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 重复键和值状态以匹配注意力头的数量
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力权重
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            else:
                causal_mask = attention_mask
            attn_weights = attn_weights + causal_mask

        # 对注意力权重进行归一化并应用dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # 确保输出的形状正确
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
                # f"`attn_output` 应该是 {(bsz, self.num_heads, q_len, self.head_dim)} 的形状, 但是得到的是 {attn_output.size()}"
            )

        # 调整形状并投影回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # 如果不需要输出注意力权重，则将其设为None
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# 从transformers.models.llama.modeling_llama.LlamaFlashAttention2复制过来，并将Llama->Gemma进行了替换
class GemmaFlashAttention2(GemmaAttention):
    """
    Gemma快速注意力模块。此模块继承自`GemmaAttention`，模块的权重保持不变。唯一需要改变的是在前向传递中正确调用flash attention的公共API，并在输入中包含填充令牌时处理填充令牌。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 一旦Flash Attention对RoCm更新到2.1版本，就应该移除这段代码。
        # flash_attn<2.1生成的是左上对齐的因果掩码，而这里需要的是右下对齐，这是flash_attn>=2.1版本中的默认行为。此属性用于处理此差异。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # 投影输入到查询、键、值空间
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash 注意力需要输入的形状为 [batch_size, sequence_length, num_heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 应用旋转位置嵌入
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        # 更新past_key_value
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # 需要静态缓存的位置信息来更新键值对
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 这些转置操作虽然效率不高，但是Flash Attention要求的布局是[batch_size, sequence_length, num_heads, head_dim]。可能需要重构KV缓存以避免这些操作。
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
                # f"输入的隐藏状态似乎被静默转换为float32，这可能与你将嵌入或层归一化层上转为float32有关。我们将输入转回{target_dtype}。"
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        调用Flash Attention的前向方法 - 如果输入的隐藏状态中至少包含一个填充令牌，首先对输入进行解填充，然后计算注意力得分，并对最终的注意力得分进行填充。

        参数:
            query_states (`torch.Tensor`): 传递给Flash Attention API的输入查询状态
            key_states (`torch.Tensor`): 传递给Flash Attention API的输入键状态
            value_states (`torch.Tensor`): 传递给Flash Attention API的输入值状态
            attention_mask (`torch.Tensor`): 填充掩码 - 对应于大小为`(batch_size, seq_len)`的张量，其中0代表填充令牌的位置，1代表非填充令牌的位置。
            dropout (`float`): 注意力dropout
            softmax_scale (`float`, *可选*): 在应用softmax之前对QK^T的缩放。默认为1 / sqrt(head_dim)
        """
        # 根据是否使用上左对齐的因果掩码来设置因果
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # 一旦Flash Attention对RoCm更新到2.1版本，应移除`query_length != 1`的检查。详情参见GemmaFlashAttention2 __init__中的评论。
            causal = self.is_causal and query_length != 1

        # 如果序列中包含至少一个填充令牌
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        """
        在序列中包含至少一个填充令牌的情况下，对输入进行解填充，然后计算注意力得分，并对最终的注意力得分进行填充。

        参数:
            query_layer (`torch.Tensor`): 查询层
            key_layer (`torch.Tensor`): 键层
            value_layer (`torch.Tensor`): 值层
            attention_mask (`torch.Tensor`): 注意力掩码
            query_length (`int`): 查询长度
        """
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 对键和值层进行索引，以匹配非填充令牌的位置
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 如果查询长度等于键值长度，则直接使用键值索引；如果查询长度为1，则处理相应地调整；否则，对查询层进行解填充
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里存在内存复制，非常不好。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 假设左填充时使用-q_len:切片
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# 从transformers.models.llama.modeling_llama.LlamaSdpaAttention复制过来，Llama->Gemma
class GemmaSdpaAttention(GemmaAttention):
    """
    使用torch.nn.functional.scaled_dot_product_attention的Gemma注意力模块。该模块继承自`GemmaAttention`，
    权重保持不变。唯一的变化是在前向传递中适配到SDPA API。
    """

    # 忽略复制
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # 提醒用户注意力输出不被支持，需要回退到手动实现。
            logger.warning_once(
                "GemmaModel is using GemmaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                # Gemma模型使用GemmaSdpaAttention，但torch.nn.functional.scaled_dot_product_attention不支持输出注意力。将回退到手动注意力实现，但从Transformers版本v5.0.0开始将需要指定手动实现。使用`attn_implementation="eager"`参数加载模型时，可以移除此警告。
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

# 定义不同的注意力类实现
GEMMA_ATTENTION_CLASSES = {
    "eager": GemmaAttention,  # 积极实现
    "flash_attention_2": GemmaFlashAttention2,  # Flash注意力2
    "sdpa": GemmaSdpaAttention,  # 标量点积注意力
}


# 从transformers.models.llama.modeling_llama.LlamaDecoderLayer复制，Llama->GEMMA
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size  # 定义隐藏层大小

        # 根据配置选择相应的注意力实现，初始化自注意力模块
        self.self_attn = GEMMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)  # 初始化多层感知机
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层正则化
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力层后的正则化

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
                # 传递`padding_mask`已被弃用，将在v4.37中删除。请确保改为使用`attention_mask`。
            )

        residual = hidden_states  # 保留输入状态作为残差连接

        hidden_states = self.input_layernorm(hidden_states)  # 对输入状态应用层正则化

        # 自注意力层处理
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states  # 残差连接

        # 全连接层处理
        residual = hidden_states  # 更新残差连接
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力层后的正则化
        hidden_states = self.mlp(hidden_states)  # 多层感知机处理
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)  # 构建输出

        if output_attentions:  # 如果需要输出注意力权重
            outputs += (self_attn_weights,)

        if use_cache:  # 如果使用缓存机制
            outputs += (present_key_value,)

        return outputs



GEMMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GemmaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
"""
    该模型继承自[`PreTrainedModel`]。请查阅超类文档了解该库为所有模型实现的通用方法（如下载或保存、调整输入嵌入的大小、修剪头部等）。

    该模型也是PyTorch的[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)子类。将其作为常规PyTorch模块使用，并参考PyTorch文档了解所有与一般使用和行为相关的事项。

    参数:
        config ([`GemmaConfig`]):
            模型配置类，包含模型的所有参数。使用配置文件进行初始化不会加载与模型相关的权重，只加载配置。查看[`~PreTrainedModel.from_pretrained`]方法来加载模型权重。
"""


# 添加文档字符串："Gemma模型的基础版本，只输出原始隐藏状态，不在顶部添加任何特定的头部。"
@add_start_docstrings(
    "The bare Gemma Model outputting raw hidden-states without any specific head on top.",
    GEMMA_START_DOCSTRING,
)
class GemmaPreTrainedModel(PreTrainedModel):
    config_class = GemmaConfig  # 模型配置类
    base_model_prefix = "model"  # 基础模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _keep_in_fp32_modules = ["inv_freq", "rotary_emb", "cos_cached", "sin_cached"]  # 保持为fp32的模块
    _no_split_modules = ["GemmaDecoderLayer"]  # 不拆分的模块
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]  # 跳过设备放置的键
    _supports_flash_attn_2 = True  # 支持Flash Attention 2
    _supports_sdpa = True  # 支持SDPA
    _supports_cache_class = True  # 支持缓存类

    def _init_weights(self, module):
        # 初始化权重
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not_none:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        # 设置缓存
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the meantime, and open an issue at https://github.com/huggingface/transformers"
                # "`static`缓存实现与`attn_implementation==flash_attention_2`不兼容，请确保在此期间使用`sdpa`，并在https://github.com/huggingface/transformers上提出问题"
            )

        if max_cache_len > self.model.causal_mask.shape[-1] or self.device != self.model.causal_mask.device:
            causal_mask = torch.full((max_cache_len, max_cache_len), fill_value=1, device=self.device)
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        for layer in self.model.layers:
            weights = layer.self_attn.o_proj.weight
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=weights.device, dtype=weights.dtype
            )

    def _reset_cache(self):
        # 重置缓存
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


GEMMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

"""
    参数:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列令牌在词汇表中的索引。如果您提供了填充，将默认忽略填充。

            索引可以使用[`AutoTokenizer`]获得。详情见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]。

            [什么是输入ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *可选*):
            避免在填充令牌索引上执行注意力的掩码。掩码值在`[0, 1]`中选择：

            - 1表示**未被掩盖**的令牌，
            - 0表示**被掩盖**的令牌。

            [什么是注意力掩码？](../glossary#attention-mask)

            索引可以使用[`AutoTokenizer`]获得。详情见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]。

            如果使用了`past_key_values`，可选地只需输入最后的`input_ids`（见`past_key_values`）。

            如果您想更改填充行为，您应阅读[`modeling_opt._prepare_decoder_attention_mask`]并根据需要进行修改。更多信息见[论文](https://arxiv.org/abs/1910.13461)中的图1。

            - 1表示头部**未被掩盖**，
            - 0表示头部**被掩盖**。
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *可选*):
            每个输入序列令牌在位置嵌入中的索引。选择范围`[0, config.n_positions - 1]`内的索引。

            [什么是位置ID？](../glossary#position-ids)
        past_key_values (`Cache`或`tuple(tuple(torch.FloatTensor))`, *可选*):
            用于加速顺序解码的预计算隐藏状态（自注意力块和交叉注意力块中的键和值）。这通常包含模型在解码的前一个阶段返回的`past_key_values`，当`use_cache=True`或`config.use_cache=True`时。

            允许两种格式：
            - 一个[`~cache_utils.Cache`]实例；
            - 一个长度为`config.n_layers`的`tuple(torch.FloatTensor)`的元组，每个元组有2个形状为`(batch_size, num_heads, sequence_length, embed_size_per_head)`的张量）。这也被称为遗留缓存格式。

            如果没有传递`past_key_values`，则会返回遗留缓存格式。

            如果使用了`past_key_values`，用户可以选择只输入最后的`input_ids`（这些`input_ids`没有给该模型的过去键值状态）的形状为`(batch_size, 1)`，而不是所有的`input_ids`的形状为`(batch_size, sequence_length)`。
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *可选*):
            可选地，而不是传递`input_ids`，您可以选择直接传递一个嵌入表示。这在您希望对如何将`input_ids`索引转换为关联向量有更多控制权时非常有用，超出了模型内部嵌入查找矩阵的能力。
        use_cache (`bool`, *可选*):
            如果设置为`True`，将返回`past_key_values`键值状态，可用于加速解码（见`past_key_values`）。
        output_attentions (`bool`, *可选*):
            是否返回所有注意力层的注意力张量。有关更多细节，请参阅返回张量下的`attentions`。
        output_hidden_states (`bool`, *可选*):
            是否返回所有层的隐藏状态。有关更多细节，请参阅返回张量下的`hidden_states`。
        return_dict (`bool`, *可选*):
            是否返回一个[`~utils.ModelOutput`]而不是一个简单的元组。
"""


@add_start_docstrings(
    "裸露的Gemma模型在顶部没有任何特定头部直接输出原始隐藏状态。",
    GEMMA_START_DOCSTRING,
)
# 从transformers.models.llama.modeling_llama.LlamaModel复制，将LLAMA更改为GEMMA
class GemmaModel(GemmaPreTrainedModel):
    """
    Transformer解码器由*config.num_hidden_layers*层组成。每一层都是一个[`GemmaDecoderLayer`]

    参数:
        config: GemmaConfig 配置类，包含模型的所有参数。
    """

    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id  # 填充令牌的索引
        self.vocab_size = config.vocab_size  # 词汇表大小

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)  # 词嵌入层
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]  # 解码器层
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最后的规范化层
        self.gradient_checkpointing = False  # 梯度检查点开关

        # 注册一个因果掩码来分离因果和填充掩码的创建。在注意力类中进行合并。
        # 注意：这不利于TorchScript、ONNX、ExportedProgram的序列化，对于非常大的`max_position_embeddings`。
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings), fill_value=True, dtype=torch.bool
        )
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)  # 注册因果掩码
        # 初始化权重并进行最后处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 根据配置决定输出注意力、隐藏状态、使用缓存和返回字典的选项

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                # "不能同时指定input_ids和inputs_embeds，必须指定其中之一"
            )

        # 如果开启了梯度检查点且正在训练且使用缓存，则警告不兼容并关闭使用缓存
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                # "`use_cache=True`与梯度检查点不兼容。设置`use_cache=False`。"
            )
            use_cache = False

        # 如果没有提供inputs_embeds，则通过词嵌入层获取
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # 为了向后兼容（缓存位置）
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        # 如果没有提供cache_position，则创建一个表示过去看到的令牌的范围
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # 如果没有提供position_ids，则使用cache_position作为position_ids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 更新因果掩码
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # 嵌入位置
        hidden_states = inputs_embeds

        # 规范化
        hidden_states = hidden_states * (self.config.hidden_size**0.5)

        # 解码器层
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # 遍历解码器层，进行前向传播
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # 如果启用了梯度检查点和训练，则使用梯度检查点功能
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                # 正常的前向传播
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 对最后的隐藏状态进行规范化
        hidden_states = self.norm(hidden_states)

        # 如果输出隐藏状态，则添加最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor):
        # 如果配置中指定使用"flash_attention_2"实现，并且提供了注意力掩码且其中包含0.0，则直接返回注意力掩码
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        # 如果序列长度超过了因果掩码的大小，则创建一个新的更大的因果掩码
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full((2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]), fill_value=1)
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        # 使用当前的数据类型来避免任何溢出
        min_dtype = torch.finfo(dtype).min

        causal_mask = self.causal_mask[None, None, :, :].to(dtype=dtype, device=device) * min_dtype
        causal_mask = causal_mask.expand(batch_size, 1, -1, -1)
        if attention_mask is not None and attention_mask.dim() == 2:
            causal_mask = causal_mask.clone()  # 将因果掩码复制到连续内存中以便进行就地编辑
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)  # 生成填充掩码，用于识别哪些位置是填充位置
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)  # 将填充位置在因果掩码中对应的值设为最小的数据类型值

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # TODO: 对于动态编译，更好的做法是一旦这成为可能，使用`fullgraph=True`的检查（https://github.com/pytorch/pytorch/pull/120400）。
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # 对于所有在因果掩码中完全被遮盖的行，比如使用左侧填充时的相关首行，这是F.scaled_dot_product_attention内存高效注意力路径所需的。
                # 详情见：https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)  # 更新因果掩码，以确保完全被遮盖的行可以正确地参与到注意力计算中

        return causal_mask


class GemmaForCausalLM(GemmaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GemmaModel(config)  # 初始化Gemma模型
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 定义语言模型头部

        self.post_init()  # 初始化权重并进行最终处理

    def get_input_embeddings(self):
        return self.model.embed_tokens  # 获取输入嵌入

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value  # 设置输入嵌入

    def get_output_embeddings(self):
        return self.lm_head  # 获取输出嵌入

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings  # 设置输出嵌入

    def set_decoder(self, decoder):
        self.model = decoder  # 设置解码器

    def get_decoder(self):
        return self.model  # 获取解码器

    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        # 根据配置决定输出格式
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 从基础模型中获取输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]  # 获取隐藏状态
        logits = self.lm_head(hidden_states)  # 通过语言模型头部计算对数几率
        logits = logits.float()  # 转换为float类型以防止精度问题
        loss = None
        if labels is not None:
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()  # 移动对数几率以便计算损失
            shift_labels = labels[..., 1:].contiguous()  # 移动标签
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            shift_logits = shift_logits.view(-1, self.config.vocab_size)  # 重新排列对数几率
            shift_labels = shift_labels.view(-1)  # 重新排列标签
            shift_labels = shift_labels.to(shift_logits.device)  # 确保标签在正确的设备上
            loss = loss_fct(shift_logits, shift_labels)  # 计算损失

        if not return_dict:
            # 如果不返回字典，构建输出元组
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not_none else output

        # 如果返回字典，构建CausalLMOutputWithPast对象并返回
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 初始化过去长度为0
        past_length = 0
        if past_key_values is not None:
            # 处理缓存键值对，计算已处理的token长度和缓存长度
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            # 根据过去长度调整input_ids和attention_mask
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            # 如果即将超出最大缓存长度，需要裁剪输入的注意力掩码
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        # 根据需要生成位置id
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 处理静态缓存的情况
        if self.generation_config.cache_implementation == "static":
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        position_ids = position_ids.contiguous() if position_ids is not None else None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # 如果提供了inputs_embeds，并且没有past_key_values，则只在第一步生成时使用inputs_embeds
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        # 更新模型输入
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 为beam搜索重新排序缓存
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The Gemma Model transformer with a sequence classification head on top (linear layer).

    [`GemmaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GEMMA_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with LLAMA->GEMMA,Llama->Gemma
# 定义一个基于Gemma预训练模型的序列分类模型
class GemmaForSequenceClassification(GemmaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)  # 初始化父类
        self.num_labels = config.num_labels  # 设置标签的数量
        self.model = GemmaModel(config)  # 初始化Gemma模型
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)  # 定义一个线性层用于分类得分

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入嵌入
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果提供了标签，说明是在计算序列分类/回归损失
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取transformer模型的输出
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # 获取隐藏状态
        logits = self.score(hidden_states)  # 计算分类得分

        # 检查是否提供了input_ids，如果提供，则使用其批量大小；否则使用inputs_embeds的批量大小
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # 如果没有定义pad_token_id且批量大小大于1，则抛出错误，因为在没有填充令牌的情况下无法处理多个序列
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        # 如果没有pad_token_id，sequence_lengths设置为-1，表示没有有效的序列长度计算方法
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            # 如果提供了input_ids和pad_token_id，计算每个序列实际长度（忽略填充部分）
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]  # 使用模运算保证索引有效性和ONNX兼容性
                sequence_lengths = sequence_lengths.to(logits.device)  # 确保序列长度在正确的设备上
            else:
                sequence_lengths = -1

        # 根据计算出的序列长度，从logits中选取对应的输出进行分类
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        # 初始化loss为None，后续根据是否提供labels进行更新
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)  # 确保标签在正确的设备上
            # 如果没有明确定义问题类型，根据num_labels和标签类型推断问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()  # 均方误差损失用于回归
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()  # 交叉熵损失用于单标签分类
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()  # 二进制交叉熵损失用于多标签分类
                loss = loss_fct(pooled_logits, labels)

        # 如果不返回字典格式，构建输出元组
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典格式，构建并返回一个具有损失、logits、过去的键值、隐藏状态和注意力的SequenceClassifierOutputWithPast对象
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


