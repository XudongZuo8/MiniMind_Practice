from transformers import PretrainedConfig
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
from transformers.activations import ACT2FN

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

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

def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return(
        x[:,:,:,None,:]
        .expand(bs,slen,num_key_value_heads,n_rep,head_dim)
        .reshape(bs,slen,num_key_value_heads*n_rep, head_dim)
    )

    
class Attention(nn.Module):
    def __init__(self,args:MiniMindConfig):
        super().__init__()
        
        self.num_key_value_heads = args.num_attention_heads\
        if args.num_key_value_heads is None else args.num_key_value_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0,\
        "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads=args.num_attention_heads #Q
        self.num_key_value_heads=args.num_key_value_heads
        self.n_rep=self.n_local_heads // self.num_key_value_heads
        self.head_dim=args.hidden_size//args.num_attention_heads

        self.q_proj=nn.linear(args.hidden_size,args.num_attention_heads*self.head_dim,
                              bias=False)
        self.k_poj=nn.linear(args.hidden_size,self.num_key_value_heads*self.head_dim,
                              bias=False)
        self.v_proj=nn.linear(args.hidden_size,self.num_key_value_heads*self.head_dim,
                              bias=False)
        self.o_proj=nn.linear(args.num_attention_heads*self.head_dim,args.hidden_size,
                              bias=False)
        
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout
        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attention

    def forward(self,x:torch.Tensor,
                position_embedding:tuple,
                past_key_value:Tuple[torch.Tensor,torch.Tensor],
                use_cache=False,
                attn_mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        # 投影，计算qkv
        bsz, seq_len, _ = x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)

        # 把输入拆分成多个头，用view
        xq = xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        xk = xk.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        xv = xv.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)

        #qk,使用rope
        cos,sin = position_embedding
        xq,xk = apply_rotary_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
         
        # 对于kv，用repeat（注意kvcache）
        if past_key_value is not None: #有kvcache
            xk = torch.cat([past_key_value[0],xk],dim=1)
            xv = torch.cat([past_key_value[1],xv],dim=1)
        past_kv = (xk,xv)

        xq,xk,xv=(
            xq.transpose(1,2),
            # [bsz,n_local,seq_len,head_dim]
            repeat_kv(xk,self.n_rep).transpose(1,2),
            repeat_kv(xv,self.n_rep).transpose(1,2)
        )
        # 进行attention计算，q@k^T/sqrt(d)
        if self.flash and (seq_len>1) and (past_key_value is None) and (attn_mask is None or torch.all
                                         (attn_mask==1)):
            output=F.scaled_dot_product_attention(xq,xk,xv,dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores=(xq@xk.transpose(-2,-1)/math.sqrt(self.head_dim))
            scores[:,:,:,-seq_len:] += torch.triu(torch.full((seq_len,seq_len),float('-inf'),device=scores.device),diagonal=1)
            if attn_mask is not None:
                extended_attention_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0-extended_attention_mask)* -1e9 # 若输入序列中有0，那么将权重压缩
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(),dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 最后拼接头
        output = output.transpose(1,2).reshape(bsz,seq_len,-1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    
class FeedFward(nn.Module):
    # initialize
    def __init__(self,args:MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size=int(args.hidden_size*8/3)
            args.intermediate_size=64*(intermediate_size+64-1//64)

        config = args.config
        self.up_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.down_proj=nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
        self.gate_proj=nn.Linear(args.hidden_size,args.intermediate_size,bisa=False)
        self.dropout=nn.Dropout(args.dropout)
        self.act_fn=ACT2FN[args.hidden_act]

    def forward(self,x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))
    







        







