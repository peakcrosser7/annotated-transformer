import copy
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import log_softmax


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query: Tensor,
              key: Tensor,
              value: Tensor,
              mask: Tensor | None = None,
              dropout: nn.Dropout | None = None) -> tuple[Tensor, Tensor]:
    """
    Compute 'Scaled Dot Product Attention'
    
    Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V

    Args:
        query (Tensor): query tensor. shape(batch_sz,h,seq_len,d_k)
        key (Tensor): key tensor. shape(batch_sz,h,seq_len,d_k)
        value (Tensor): value tensor. shape(batch_sz,h,seq_len,d_k)
        mask (Tensor | None, optional): mask tensor. query tensor. \
            shape(1,1,1 | seq_len,seq_len). Defaults to None.
        dropout (nn.Dropout | None, optional): dropout layer. Defaults to None.

    Returns:
        tuple[Tensor, Tensor]: 
            attention output tensor. shape(batch_sz,h,seq_len,d_k). \n
            attention weights tensor(softmax(QK^T/\sqrt{d_k})). shape(batch_sz,h,seq_len,seq_len).
    """
    d_k = query.size(-1)
    # QK^T/\sqrt{d_k}
    #   shape (batch_sz,h,seq_len,d_k)*(batch_sz,h,d_k,seq_len) -> (batch_sz,h,seq_len,seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 将掩码元素替换为很小的负数-1e9
        scores = scores.masked_fill(mask == 0, -1e9)
    # 对scores最后一维进行softmax
    #   p_attn.shape(batch_sz,h,seq_len,seq_len)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # softmax(QK^T/\sqrt{d_k})*V
    #   shape (batch_sz,h,seq_len,seq_len)*(batch_sz,h,seq_len,d_k) -> (batch_sz,h,seq_len,d_k)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h: int, d_model: int, dropout=0.1):
        """Take in model size and number of heads.

        Args:
            h (int): number of multi-heads
            d_model (int): dimension of the model
            dropout (float, optional): Dropout value. Defaults to 0.1.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 4个d_model×d_model的线性层
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Tensor | None = None) -> Tensor:
        """
        Implements Figure 2

        MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O, \
            where head_i=Attention(QW^Q_i,KW^K_i,VW^V_i)

        Args:
            query (Tensor): query tensor. shape(batch_sz,seq_len,d_model)
            key (Tensor): key tensor. shape(batch_sz,seq_len,d_model)
            value (Tensor): value tensor. shape(batch_sz,seq_len,d_model)
            mask (Tensor | None, optional): mask tensor. query tensor. \
                shape(1,1 | seq_len,seq_len). Defaults to None.

        Returns:
            Tensor: multi-head attention output tensor. shape(batch_sz,seq_len,d_model)
        """
        if mask is not None:
            # Same mask applied to all h heads.
            # 在1维度上插入一个新的维度
            #   shape (1,1 | seq_len,seq_len) -> (1,1,1 | seq_len,seq_len)
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 对于Q,K,V使用各自的线性层(linears[0:3] W^Q,W^K,W^V进行变化)
        query, key, value = [
            # lin(x):经过线性变换Ax+b(实际是x*A^T+b)
            #   shape (batch_sz,seq_len,d_model) -> (batch_sz,seq_len,d_model)
            # view():重塑张量形状
            #   shape (batch_sz,seq_len,d_model) -> (batch_sz,seq_len,h,d_k)
            # transpose():转置1,2维度
            #   shape (batch_sz,seq_len,h,d_k) -> (batch_sz,h,seq_len,d_k)
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            # for循环只迭代3次,每次迭代x依次表示query,key,value
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        #   x.shape(batch_sz,h,seq_len,d_k)
        #   attn.shape(batch_sz,h,seq_len,seq_len)
        x, self.attn = attention(query,
                                 key,
                                 value,
                                 mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # transpose(1, 2): 转置1和2维度
        #   shape (batch_sz,h,seq_len,d_k) -> (batch_sz,seq_len,h,d_k)
        # contiguous(): 将转置后的张量内存连续化(以便使用`view()`)
        # view(): 重塑张量形状
        #   shape (batch_sz,seq_len,h,d_k) -> (batch_sz,seq_len,h*d_k)
        x = (x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                 self.h * self.d_k))
        del query
        del key
        del value
        # 经过最后一次线性层W^O变化
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        """
        Args:
            d_model (int): dimension of the model
            d_ff (int): inner-layer dimension between two linear layers
            dropout (float, optional): Dropout value. Defaults to 0.1.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        FFN(x)=Relu(xW_1+b_1)W_2+b_2

        Args:
            x (Tensor): input tensor. shape(batch_sz,seq_len,d_model)

        Returns:
            Tensor: output tensor. shape(batch_sz,seq_len,d_model)
        """
        # shape (batch_sz,seq_len,d_model)*(d_model,d_ff) -> (batch_sz,seq_len,d_ff)
        # shape (batch_sz,seq_len,d_ff)*(d_ff,d_model) -> (batch_sz,seq_len,d_model)
        return self.w_2(self.dropout(self.w_1(x).relu()))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features: int, eps=1e-6):
        """
        Args:
            features (int): feature dimension of the input
            eps (float, optional):  a value added to the denominator for numerical stability. Defaults to 1e-6.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # shape(d_model)
        self.b_2 = nn.Parameter(torch.zeros(features))  # shape(d_model)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        y_i=alpha * (x_i-\mu)/(\sigma+\epsilon) + beta

        Args:
            x (Tensor): input tensor. shape(batch_sz,seq_len,d_model)

        Returns:
            Tensor: output tensor. shape(batch_sz,seq_len,d_model)
        """
        # 最后一维求均值\mu
        #   shape(batch_sz,seq_len,1)
        mean = x.mean(-1, keepdim=True)
        # 最后一维求标准差
        #   shape(batch_sz,seq_len,1)
        std = x.std(-1, keepdim=True)
        # y_i=\alpha \frac{x_i-\mu}{\sigma+\epsilon}+\beta
        #   shape(batch_sz,seq_len,d_model)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float):
        """
        Args:
            size (int): size(dimension) of the layer
            dropout (float): Dropout value
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Tensor) -> Tensor:
        """
        Apply residual connection to any sublayer with the same size.

        x+Sublayer(LayerNorm(x))

        Args:
            x (Tensor): input embeddings tensor. shape(batch_sz,seq_len,d_model)
            sublayer (Tensor):  sub-layer output tensor. shape(batch_sz,seq_len,d_model)

        Returns:
            Tensor: output tensor after a residual connection and layer normalization.\
                shape(batch_sz,seq_len,d_model)
        """
        # 此处使用了pre-Norm,与Transformer论文中的post-Norm不同
        # 参考: https://zhuanlan.zhihu.com/p/474988236
        # post-Norm: LayerNorm(x+Sublayer(x))
        # pre-Norm: x+Sublayer(LayerNorm(x))
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size: int, self_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float):
        """
        Args:
            size (int): size(dimension) of the layer
            self_attn (MultiHeadedAttention): self-attention network
            feed_forward (PositionwiseFeedForward): position-wise feed forward network
            dropout (float): Dropout value
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Follow Figure 1 (left) for connections.

        Multi-Head Attention -> Add&Norm -> FeedForward -> Add&Norm

        Args:
            x (Tensor): input tensor. shape(batch_sz,seq_len,d_model)
            mask (Tensor): mask tensor. shape(1,1,seq_len)

        Returns:
            Tensor: output tensor through a encode layer. shape(batch_sz,seq_len,d_model)
        """
        # Multi-Head Attention -> Add&Norm
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # FeedForward -> Add&Norm
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer: EncoderLayer, N: int):
        """
        Args:
            layer (EncoderLayer): an encoder layer
            N (int): number of encoder layers
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Pass the input (and mask) through each layer of Encoder in turn.

        Args:
            x (Tensor): input embeddings. shape(batch_sz,seq_len,d_model)
            mask (Tensor): mask tensor. shape(1,1,seq_len)

        Returns:
            Tensor: output tensor through the Encoder. shape(batch_sz,seq_len,d_model)
        """
        # 经过N个EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size: int, self_attn: MultiHeadedAttention,
                 src_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float):
        """
        Args:
            size (int): size(dimension) of the layer
            self_attn (MultiHeadedAttention): self-attention network
            src_attn (MultiHeadedAttention): source-attention network
            feed_forward (PositionwiseFeedForward): position-wise feed forward network
            dropout (float): Dropout value
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor,
                tgt_mask: Tensor) -> Tensor:
        """
        Follow Figure 1 (right) for connections.

        Args:
            x (Tensor): input tensor. shape(batch_sz,seq_len,d_model) 
            memory (Tensor): the memory(output tensor) of the Encoder. shape(batch_sz,seq_len,d_model)
            src_mask (Tensor): mask of source sequences. shape(1,1,seq_len)
            tgt_mask (Tensor): mask of target sequences. shape(1,seq_len,seq_len)

        Returns:
            Tensor: output tensor through a decoder layer. shape(batch_sz,seq_len,d_model )
        """
        m = memory
        # Masked Multi-Head Attention -> Add&Norm (Q,K,V来自上一层)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Multi-Head Attention -> Add&Norm (Q来自上一层,K,V来自Encoder的memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # FeedForward -> Add&Norm
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: DecoderLayer, N: int):
        """
        Args:
            layer (DecoderLayer): an decoder layer
            N (int): number of decoder layers
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor,
                tgt_mask: Tensor) -> Tensor:
        """Pass the input (and mask) through each layer of Decoder in turn.

        Args:
            x (Tensor): input embeddings of target sequences. shape(batch_sz,seq_len,d_model) 
            memory (Tensor): the memory(output tensor) of the Encoder. shape(batch_sz,seq_len,d_model)
            src_mask (Tensor): mask of source sequences. shape(1,1,seq_len)
            tgt_mask (Tensor): mask of target sequences. shape(1,seq_len,seq_len)

        Returns:
            Tensor: output tensor through the Decoder. shape(batch_sz,seq_len,d_model)
        """
        # 经过N个DecoderLayer
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Embeddings(nn.Module):

    def __init__(self, d_model: int, vocab: int):
        """
        Args:
            d_model (int): dimension of the model
            vocab (int): length of vocabularies
        """
        super(Embeddings, self).__init__()
        # 嵌入表大小为vocab嵌入维度为d_model的查询表(lookup table)
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """generate embeddings of input sequences

        Args:
            x (Tensor): input sequences. shape(batch_sz,seq_len)

        Returns:
            tensor: embeddings of input sequences. shape(batch_sz,seq_len,d_model)
        """
        # lut(x): 经过查询表得到序列对应的嵌入 shape(batch_sz,seq_len,d_model)
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model: int, dropout: float, max_len=5000):
        """
        Args:
            d_model (int): dimension of the model
            dropout (float): Dropout value
            max_len (int, optional): maximum length of the sequence. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # 位置编码张量 shape(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 创建一个值为0~(max_len-1)的张量并在1维度扩展一维
        # unsqueeze(1): shape (max_len,) -> (max_len,1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # e^(2i * -ln(10000)/d_model)= 1/10000^(2i/d_model)
        div_term = torch.exp(
            # arange(): 创建0~(d_model-1)步长为2的张量 shape(d_model//2,)
            # 2i * -ln(10000)/d_model
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # pe偶数列(1维度从0开始,步长为2): sin(pos/10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe奇数列: cos(pos/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        # 在0维度扩展一维: shape (max_len,d_model) -> (1,max_len,d_model)
        pe = pe.unsqueeze(0)
        # 将pe作为常量注册到模型的状态表中
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """add "position encodings" to the input embeddings

        Args:
            x (Tensor): input embeddings. shape(batch_sz,seq_len,d_model)

        Returns:
            Tensor: input embeddings with position encodings. shape(batch_sz,seq_len,d_model)
        """
        # x.size(1)为序列长度seq_len
        # self.pe[:, : x.size(1)]: 在1维度切片pe shape(1,seq_len,d_model)
        # requires_grad_(False): pe已经作为常量注册到状态表中,本身requires_grad属性就是False.可以认为这一步是多余的.
        # 利用Pytorch广播机制,对batch_sz个sequences都加上pe的位置编码
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model: int, vocab: int):
        """
        Args:
            d_model (int): dimension of the model
            vocab (int): length of source vocabularies
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        """generate the ouput probabilities

        Args:
            x (Tensor): input tensor. shape(batch_sz,d_model)

        Returns:
            Tensor: ouput probabilities. shape(batch_sz,vocab)
        """
        # proj(x): 线性映射到单词表维度 
        #   shape (batch_sz,d_model) -> (batch_sz,vocab)
        return log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: nn.Sequential, tgt_embed: nn.Sequential,
                 generator: Generator):
        """
        Args:
            encoder (Encoder): the encoder
            decoder (Decoder): the decoder 
            src_embed (nn.Sequential): sequential container for embedding souce sequences. \
                Including `Embeddings` and `PositionalEncoding` two models.
            tgt_embed (nn.Sequential): sequential container for embedding target sequences. \
                Including `Embeddings` and `PositionalEncoding` two models.
            generator (Generator): the generator for target sequences
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Take in and process masked src and target sequences.

        Args:
            src (Tensor): source sequences to be encoded. shape(batch_sz,seq_len)
            tgt (Tensor): target sequences to be decoded. shape(batch_sz,seq_len)
            src_mask (Tensor): mask of source sequences. shape(1,1,seq_len)
            tgt_mask (Tensor): mask of target sequences. shape(1,seq_len,seq_len)

        Returns:
            Tensor: output tensor through the Encoder and Decoder of Transformer. \
                shape(batch_sz,seq_len,d_model)
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """encode a sequence by the Encoder of Transformer

        Args:
            src (Tensor): source sequences to be encoded. shape(batch_sz,seq_len)
            src_mask (Tensor): mask of source sequences. shape(1,1,seq_len)

        Returns:
            Tensor: output tensor through the Encoder. shape(batch_sz,seq_len,d_model)
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: Tensor, src_mask: Tensor, tgt: Tensor,
               tgt_mask: Tensor) -> Tensor:
        """decode a sequence by the Decoder of Transformer

        Args:
            memory (Tensor): the memory(output tensor) of the Encoder. shape(batch_sz,seq_len,d_model)
            src_mask (Tensor): mask of source sequences. shape(1,1,seq_len)
            tgt (Tensor): target sequences to be decoded. shape(batch_sz,seq_len)
            tgt_mask (Tensor): mask of target sequences. shape(1,seq_len,seq_len)

        Returns:
            Tensor: output tensor through the Decoder. shape(batch_sz,seq_len,d_model)
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab: int,
               tgt_vocab: int,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1) -> EncoderDecoder:
    """Helper: Construct a model from hyperparameters.

    Args:
        src_vocab (int): length of source vocabularies
        tgt_vocab (int): length of target vocabularies
        N (int, optional): number of layers of Encoder and Decoder. Defaults to 6.
        d_model (int, optional): dimension of the model. Defaults to 512.
        d_ff (int, optional): inner-layer dimension of Feed-Forward Network. Defaults to 2048.
        h (int, optional): number of multi-heads. Defaults to 8.
        dropout (float, optional): Dropout value, i.e. probability of an element to be zeroed. Defaults to 0.1.

    Returns:
        EncoderDecoder: a model of Transformer
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 使用Xavier均匀分布初始化模型的权重参数
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size: int) -> Tensor:
    """Mask out subsequent positions.

    Args:
        size (int): length of the mask

    Returns:
        Tensor: mask of subsequent positions. shape(1,size,size)
    """
    attn_shape = (1, size, size)
    # triu(): 生成一个上三角矩阵作为掩码矩阵, diagonal为1表示主对角线以上1个偏移的对角线
    #   shape(1,size,size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    # 将掩码矩阵中的1变为0,0变为1
    return subsequent_mask == 0
