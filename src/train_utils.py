import time

import typing
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


from model import *

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class LabelSmoothing(nn.Module):
    """
    Regularization strategy: label smoothing.
    """

    def __init__(self, size: int, padding_idx: int, smoothing=0.0):
        """
        Args:
            size (int): the size of vocabularies
            padding_idx (int): index representing padding
            smoothing (float, optional): smoothing factor. Defaults to 0.0.
        """
        super(LabelSmoothing, self).__init__()
        # Kullback-Leibler散度损失对象,损失函数的降维方式(求和)
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        # 置信度,即非平滑部分的权重
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): output tensor from the generator of the model. \
                shape(batch_sz*seq_len,vocab)
            target (Tensor): correct target sequences. shape(batch_sz*seq_len)

        Returns:
            Tensor: loss value based on label smoothing. shape()
        """
        assert x.size(1) == self.size
        # shape(batch_sz*seq_len,vocab)
        true_dist = x.data.clone()
        # 就地赋值
        true_dist.fill_(self.smoothing / (self.size - 2))
        # scatter_(dim, index, src):将源张量src按照索引张量index在维度dim上散射
        # target.data.unsqueeze(1): shape(batch_sz*seq_len,1)
        # 将true_dist的1维度上按照target作为索引赋值为confidence
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 将填充token置0
        true_dist[:, self.padding_idx] = 0
        # 返回target中值为padding_idx的索引
        #   shape(nnz,*target.data.shape)=(nnz,batch_sz*seq_len)
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # index_fill_(dim,index,value):在维度dim上按照张量index赋值value
            # 在0维度上,将mask中为padding_idx的位置置零
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # 计算损失
        return self.criterion(x, true_dist.clone().detach())


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src: Tensor, tgt: Tensor | None = None, pad = 2):  # 2 = <blank>
        """
        Args:
            src (Tensor): source sequences. shape(batch_sz,seq_len)
            tgt (Tensor | None, optional): target sequences. Defaults to None. shape(batch_sz,seq_len)
            pad (int, optional): blank padding. Defaults to 2.
        """
        # 源序列
        self.src = src
        # 逐元素比较,并在倒数第2维新增一维
        #   shape (batch_sz,seq_len) -> (batch_sz,1,seq_len)
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # 切片除最后一列得到目标序列的前缀,作为模型的输入目标序列(用于进行预测)
            #   shape(batch_sz,seq_len-1)
            self.tgt = tgt[:, :-1]
            # 切片除第一列得到目标序列的后缀,作为模型的正确输出结果(预测的正确结果)
            #   shape(batch_sz,seq_len-1)
            self.tgt_y = tgt[:, 1:]
            # 目标序列掩码 shape(batch_sz,seq_len,seq_len)
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # 求和得到总token数 标量张量shape()
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt: Tensor, pad: int) -> Tensor:
        """Create a mask to hide padding and future words.

        Args:
            tgt (Tensor): target sequences. shape(batch_sz,seq_len)
            pad (int): blank padding

        Returns:
            Tensor: a mask to hide padding and future words. shape(batch_sz,seq_len,seq_len)
        """
        # 将不为空白填充的字符记为True
        #   shape (batch_sz,seq_len) -> (batch_sz,1,seq_len)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # &操作:结合填充字符和后续字符遮盖得到新的掩码.
        #   运算时利用广播机制,1维度上会复制seq_len次2维度数据
        #   shape (batch_sz,1,seq_len) & (1,seq_len,seq_len) -> (batch_sz,seq_len,seq_len)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask     # shape(batch_sz,seq_len,seq_len)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator: Generator, criterion: LabelSmoothing):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: Tensor, y: Tensor, norm: Tensor) -> tuple[Tensor, Tensor]:
        """compute loss

        Args:
            x (Tensor): output tensor from the decoder of the model. \
                shape(batch_sz,seq_len,d_model)
            y (Tensor): correct target sequences. shape(batch_sz,seq_len)
            norm (Tensor): total number of tokens in the sequences. shape()

        Returns:
            tuple[Tensor, Tensor]: 
                Tensor: total loss. shape()
                Tensor: mean loss. shape()
        """
        # 生成对应的目标序列的概率
        #   shape (batch_sz,seq_len,d_model) -> (batch_sz,seq_len,vocab)
        x = self.generator(x)
        # 标量 shape()
        sloss = (
            # 标量 shape()
            self.criterion(
                # x.shape(batch_sz*seq_len,vocab)
                # y.shape(batch_sz*seq_len)
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.

    Args:
        step (int): number of the epoch
        model_size (int): size(dimension) of the model
        factor (float): a factor multiplied on the calculation result
        warmup (int): number of warpup steps

    Returns:
        float: learning rate
    """
    if step == 0:
        step = 1
    return factor * (
        # lrate=d_model^{-0.5}*min(step_num^{-0.5}, step_num*warpup_steps^{-1.5})
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def greedy_decode(model: EncoderDecoder, src: Tensor, src_mask: Tensor, max_len: int, start_symbol: int) -> Tensor:
    """predict a translation using greedy decoding

    Args:
        model (EncoderDecoder): Transformer model
        src (Tensor): source sequences to be encoded. shape(batch_sz,seq_len)
        src_mask (Tensor): mask of source sequences. shape(1,1,seq_len)
        max_len (int): maxium length of the output sequence
        start_symbol (int): start symbol(token) in the output sequence

    Returns:
        Tensor: output predicted sequence. shape(1,max_len)
    """
    # 经过Transformer编码源序列得到记忆值
    #   shape(batch_sz,seq_len,d_model)
    memory = model.encode(src, src_mask)
    # 初始化输出序列
    #   shape(1,1)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    # 预测后续的max_len-1个字符
    for i in range(max_len - 1):
        # 经过Transformer解码
        #   shape(1,ys.size(1),d_model)
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # 生成最后一个时间步的字符概率
        #   shape (1,d_model) -> (1, vocab)
        prob = model.generator(out[:, -1])
        # 将概率最大的对应索引作为输出字符 shape(1)
        _, next_word = torch.max(prob, dim=1)
        # shape() 
        next_word = next_word.data[0]
        # 将next_word拼接到输出序列ys
        #   shape (1,ys.size(1)) -> (1,ys.size(1)+1)
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def run_epoch(
    data_iter: typing.Generator[Batch, None, None],
    model: EncoderDecoder,
    loss_compute: SimpleLossCompute,
    optimizer: torch.optim.Adam,
    scheduler: LambdaLR,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
) -> tuple[Tensor, TrainState]:
    """Train a single epoch

    Args:
        data_iter (typing.Generator[Batch, None, None]): a generator to generate a batch of data
        model (EncoderDecoder): Transformer model
        loss_compute (SimpleLossCompute): loss compute object
        optimizer (torch.optim.Adam): optimizer
        scheduler (LambdaLR): learning rate scheduler
        mode (str, optional): mode of the model. Defaults to "train".
        accum_iter (int, optional): number of iteration for gradient accumulation. Defaults to 1.
        train_state (TrainState, optional): an object to trace trainning state. Defaults to TrainState().

    Returns:
        tuple[Tensor, TrainState]: 
            Tensor: mean loss per token. shape()
            TrainState: an object to trace trainning state
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        # 模型前向传播
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        # 计算损失
        # 总损失和评价损失
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            # 反向传播计算张量梯度
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]   # batch_sz
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                # 根据梯度优化模型参数
                optimizer.step()
                # 清空模型梯度. 
                # `set_to_none=True`:直接将梯度设为None
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            # 调度器更新学习率
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            # 获取学习率
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state