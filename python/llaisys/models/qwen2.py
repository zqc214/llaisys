from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, llaisysQwen2Model_t

from pathlib import Path
import safetensors.torch
import ctypes
import json
import numpy as np
import torch


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 读取模型配置
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # 提取模型参数
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.intermediate_size = config["intermediate_size"]
        self.vocab_size = config["vocab_size"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.rms_norm_eps = config["rms_norm_eps"]
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.eos_token_id = config.get("eos_token_id", 151643)
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]
        
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # 创建模型元数据
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.BF16.value  # DeepSeek-R1-Distill-Qwen-1.5B 使用 bfloat16
        meta.nlayer = self.num_hidden_layers
        meta.hs = self.hidden_size
        meta.nh = self.num_attention_heads
        meta.nkvh = self.num_key_value_heads
        meta.dh = self.head_dim
        meta.di = self.intermediate_size
        meta.maxseq = self.max_position_embeddings
        meta.voc = self.vocab_size
        meta.epsilon = self.rms_norm_eps
        meta.theta = self.rope_theta
        meta.end_token = self.eos_token_id
        
        # 创建模型
        self.device = device
        device_id = 0
        device_ids = (ctypes.c_int * 1)(device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            device.value,
            device_ids,
            1
        )
        
        # 获取权重指针
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        
        # 加载权重
        self._load_weights(model_path, device)
    
    def _load_weights(self, model_path: Path, device: DeviceType):
        """从 safetensors 文件加载权重"""
        
        # 名称映射
        name_map = {
            "model.embed_tokens.weight": "in_embed",
            "lm_head.weight": "out_embed",
            "model.norm.weight": "out_norm_w",
        }
        
        layer_name_map = {
            "input_layernorm.weight": "attn_norm_w",
            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",
            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",
            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",
            "self_attn.o_proj.weight": "attn_o_w",
            "post_attention_layernorm.weight": "mlp_norm_w",
            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }
        
        print(f"[LLAISYS] Loading weights from {model_path}...")
        for file in sorted(model_path.glob("*.safetensors")):
            print(f"[LLAISYS] Loading {file.name}...")
            # 使用 torch 加载以支持 bfloat16
            data_ = safetensors.torch.load_file(file, device="cpu")
            for name_, tensor_torch in data_.items():
                # 确定权重名称和层索引
                if name_ in name_map:
                    self._set_weight(name_map[name_], -1, tensor_torch, device)
                else:
                    # 解析层号和权重名称
                    # 格式: model.layers.{layer_idx}.{weight_name}
                    parts = name_.split(".")
                    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
                        layer_idx = int(parts[2])
                        weight_name = ".".join(parts[3:])
                        if weight_name in layer_name_map:
                            self._set_weight(layer_name_map[weight_name], layer_idx, tensor_torch, device)
    
    def _set_weight(self, weight_name: str, layer_idx: int, tensor_torch: torch.Tensor, device: DeviceType):
        """设置单个权重张量"""
        # 确保张量是连续的
        tensor_torch = tensor_torch.contiguous()
        
        # 创建张量
        shape = tensor_torch.shape
        shape_arr = (ctypes.c_size_t * len(shape))(*shape)
        
        # 确定数据类型
        if tensor_torch.dtype == torch.float32:
            dtype = DataType.F32
        elif tensor_torch.dtype == torch.float16:
            dtype = DataType.F16
        elif tensor_torch.dtype == torch.bfloat16:
            dtype = DataType.BF16
        else:
            # 转换为 bfloat16
            tensor_torch = tensor_torch.to(torch.bfloat16)
            dtype = DataType.BF16
        
        device_id = 0
        tensor = LIB_LLAISYS.tensorCreate(
            shape_arr,
            len(shape),
            dtype.value,
            device.value,
            device_id
        )
        
        # 加载数据 - 使用 torch tensor 的数据指针
        LIB_LLAISYS.tensorLoad(tensor, ctypes.c_void_p(tensor_torch.data_ptr()))
        
        # 设置到权重结构体
        if layer_idx < 0:
            # 全局权重
            if weight_name == "in_embed":
                self._weights.in_embed = tensor
            elif weight_name == "out_embed":
                self._weights.out_embed = tensor
            elif weight_name == "out_norm_w":
                self._weights.out_norm_w = tensor
        else:
            # 层权重
            if weight_name == "attn_norm_w":
                self._weights.attn_norm_w[layer_idx] = tensor
            elif weight_name == "attn_q_w":
                self._weights.attn_q_w[layer_idx] = tensor
            elif weight_name == "attn_q_b":
                self._weights.attn_q_b[layer_idx] = tensor
            elif weight_name == "attn_k_w":
                self._weights.attn_k_w[layer_idx] = tensor
            elif weight_name == "attn_k_b":
                self._weights.attn_k_b[layer_idx] = tensor
            elif weight_name == "attn_v_w":
                self._weights.attn_v_w[layer_idx] = tensor
            elif weight_name == "attn_v_b":
                self._weights.attn_v_b[layer_idx] = tensor
            elif weight_name == "attn_o_w":
                self._weights.attn_o_w[layer_idx] = tensor
            elif weight_name == "mlp_norm_w":
                self._weights.mlp_norm_w[layer_idx] = tensor
            elif weight_name == "mlp_gate_w":
                self._weights.mlp_gate_w[layer_idx] = tensor
            elif weight_name == "mlp_up_w":
                self._weights.mlp_up_w[layer_idx] = tensor
            elif weight_name == "mlp_down_w":
                self._weights.mlp_down_w[layer_idx] = tensor
    
    def __del__(self):
        if hasattr(self, '_model') and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """生成文本"""
        if max_new_tokens is None:
            max_new_tokens = 128
        
        # 将输入转换为列表
        output_tokens = list(inputs)
        
        # 首先处理所有输入 token (prefill)
        print(f"[LLAISYS] Prefill with {len(inputs)} tokens...")
        input_arr = (ctypes.c_int64 * len(inputs))(*inputs)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model,
            input_arr,
            len(inputs)
        )
        output_tokens.append(next_token)
        print(f"[LLAISYS] Prefill done, first token: {next_token}")
        
        # 然后逐个生成新 token (decode)
        for i in range(max_new_tokens - 1):
            if next_token == self.eos_token_id:
                print(f"[LLAISYS] EOS token reached at step {i+1}")
                break
            
            # 单 token 推理
            single_token = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                single_token,
                1
            )
            output_tokens.append(next_token)
            
            if (i + 1) % 10 == 0:
                print(f"[LLAISYS] Generated {i + 2} tokens...")
        
        print(f"[LLAISYS] Generation complete, total {len(output_tokens)} tokens")
        return output_tokens
