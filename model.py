from ast import Module
# from regex import F
# from sphinx import ret
import torch 
import torch.nn as nn 
import math 
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 12 
    n_heads : int = 32
    nk_heads : Optional[int] =None
    vocab_size : int =  -1
    multiple_off : int = 256 
    ffn_dim_multiplier :    int =   1 
    norm_eqs : float = 1e-5
    
    max_batch_size : int =  32
    max_seq_length : int =2048
    
    device :str = None
def precompute_theta_pos_frequanceis(self , head_dim: int , seq_len , device :str , theta: float=10000.0 ):
    assert head_dim % 2==0 
    
    theta_numerator = torch.arange(0 , head_dim, 2 ).float()
    theta = 1 /(theta** (theta_numerator /head_dim)).to(device)
    
    m = torch.arange(seq_len , device= device )
    freqs = torch.outer(m , theta , device= device)
    freq_complex = torch.polar(freqs , torch.like_ones(freqs))
    return freq_complex

def apply_rotary_embeddings(x:torch.Tensor  ,freqs_complex: torch.Tensor , device :str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1 , -1 , 2]))
   
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_complex(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def  repeat_kv(x: torch.Tensor , n_rep :int ) -> torch.Tensor :
    batch_size , seq_len , n_kv_heads  , head_dim  = x.shape
    if n_rep == 1 :
        return x
    else : 
        return (
            x[: , : , :, None ,:]    
            .expand(batch_size , seq_len , n_kv_heads , head_dim)
            .reshape(batch_size , seq_len , n_kv_heads * n_rep , head_dim)
            
        )
     
class RMSNorm(nn.Module):
    
    def __init__(self , dim : int  , eps:float = 1e-6):
        super().__init__(self)
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self , x:torch.Tensor)  :
        return x*torch.rsqrt(x.pow(2).mean(-1 , keepdim=True)+ self.eps)
    
    def forward(self , x:torch.Tensor):
        return self.weights + self._norm(x.float()).type_as(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.atten = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim , eps = args.norm_eps)
        self.ff_norm = RMSNorm(args.dim , eps = args.norm_eps)
    def forward(self , x :torch.Tensor ,start_pos: int  , freqs_complex: torch.Tensor ):
        h = x + self.attention.forward(self.attention_norm(x) , start_pos = start_pos, freqs_complex= freqs_complex)
        out = self.feed_forward(self.ff_norm(h))
        return out
class SelfAttention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init()
        self.n_heads = args.n_heads 
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is None else args.n_kv_heads
        self.rep = self.n_heads_q // self.n_kv_heads
        self.head_dim  = args.dims // args.n_heads
        
        self.wq = nn.Linear(args.dim , args.n_heads  * self.head_dim, bias = False )
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim , args.dim , bias = False)   
                     
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_length , self.n_kv_heads , self.head_dim ))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_length , self.n_kv_heads , self.head_dim ))
        
    def forward(self , x :torch.Tensor , start_pos :int , freqs_complex ):
        
        batch_size , seq_len , _ = x.shape
        xq = self.wq(x)
        
        xv = self.wv(x)
        xk = self.wk(x)
        
        
        xq = xq.view(batch_size , seq_len , self.n_heads_q , self.head_dim)
        xk = xk.view(batch_size , seq_len , self.n_kv_heads , self.head_dim)
        xq = xq.view(batch_size , seq_len , self.n_kv_heads , self.head_dim)
        
        xq = apply_rotary_embeddings(xq, freqs_complex , device = x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex , device = x.device)
        
        self.cache_k[:batch_size , start_pos : start_pos + seq_len] = xk
        
        self.cache_v[:batch_size , start_pos : start_pos + seq_len] = xv 
         
        keys = self.cache_k[:batch_size , 0 : start_pos + seq_len]
        values = self.cache_v[:batch_size , 0 : start_pos + seq_len]  
        
        keys = repeat_kv(keys , self.n_rep) 
        values = repeat_kv(values , self.n_rep)
        
        xq = xq.transpose(1 , 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2 , 3 )) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float() , dim = -1 ).type_as(xq)        

        out = torch.matmul(scores, values)
        output = (output.transpose(1 , 2).contiguous().view(batch_size , seq_len , -1))
        
        return self.wo(output)
class FeedForward(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        hidden_dim = args.dim 
        hidden_dim = int(2*hidden_dim / 3 )
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        hidden = args.multiple_off *((hidden +args.multiple_off -1 )// args.multiple_off)    
        
        self.w1 = nn.linear(args.dim , hidden_dim , bias = False)      
        self.w2 = nn.linear(hidden_dim , args.dim , bias = False)
        self.w3 = nn.linear(args.dim , hidden_dim , bias = False)
        
    def forwrd(self , x :torch.Tensor ):
        swish = F.silu(self.w1(x))
        x_V = self.w2(x)
        x = swish * x_V
        x = self.w2(x)
        return x
        
            
    
                           
class Transformer(nn.Module):     
    def __init__(self , args , ModelArgs) -> None :
        super().__init__()
        assert args.vocab_size !=-1
        
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embedding = nn.Embedding(self.vocab_size, args.dim)
        
        self.layer = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layer.append(EncoderBlock[args])
        
        self.norm = RMSNorm (args.dim ,eps = args.eps )
        self.output= nn.Linear(args.dim , self.vocab_size, bias = False)

        self.freqs_complex = precompute_theta_pos_frequanceis(self.args.dim // self.args.n_heads , self.args.max_seq_len+2 , device = self.args.device)
    def forward(self ,tokens: torch.Tensor , start_pos :int):
        batch_size , seq_len = tokens.shape
        assert seq_len == 1
        
        h = self.tok_embedding(tokens)
        freqs_complex = self.freqs_complex[start_pos :start_pos ,+seq_len] 
        
        for layer in self.layers:
            h  =layer(h , start_pos,  freqs_complex)
        h = self.norm(h)
        output  = self.output(h).float()
        return output
