from transformers import PretrainedConfig
import math

class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
# 首先继承module
class RMSNorm(nn.Module):
    # 接着初始化init
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim)) # 最后一个参数是公式里的可优化参数
    # 接着将rmsnorm写成代码的表示方式
    def _norm(self,x):
        return torch.rsqrt((x.pow(2).mean(-1,keepdim=True)+self.eps))*x
    
    # 最后forward()
    def forward(self,x):
        return self.weight * self._norm(x.float()).type_as(x)

# 首先写出rope算式
def precomput_cis_freqs(dim:int,
                        end:int=int(32*1024),
                        rope_base:float=1e6,
                        rope_scaling:Optional[dict]=None):
    freqs = 1.0/rope_base**(torch.arrange(0,dim,2)[:dim//2].float()/dim)
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )
        #计算corr_dim
        corr_dim = next((i for i in range(dim//2) if 2*math.pi/freqs[i] >orig_max), dim//2)

        # 计算power
        power  = torch.arange(0,dim//2,device=freqs.device).float()/max(dim//2-1,1)

        #计算beta
        beta = beta_slow + (beta_fast-beta_slow)*power

        #计算scale
        scale = torch.where(
            torch.arange(0,dim//2,device=freqs.device) < corr_dim,
            (beta*factor-beta+1.0)/(beta*factor),
            1/factor
        )
        # 应用scale
        freqs = freqs*scale
        # 生成位置索引,与频率相乘，得到完整的矩阵
    t = torch.arange(end,device = freqs.device)
    freqs = torch.outer(t,freqs).float()
    # 计算cis
    # freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)
    # freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)
    freqs_cos = torch.cos(freqs).repeat_interleave(2,dim=-1)
    freqs_sin = torch.sin(freqs).repeat_interleave(2,dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1):
    #维度对齐
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x):
        # 取出偶数位：q0\q2\q4
        x1 = x[...,::2]
        # 取出奇数位：q1\q3\q5
        x2 = x[...,1::2]
        return torch.stack([-x2,x1],dim=-1).flatten(-2)
    
    #应用旋转位置编码
    q_embed = q*cos + rotate_half(q)*sin
    k_embed = k*cos + rotate_half(k)*sin
    return q_embed, k_embed

