from train_utils import *


def data_gen(V: int, batch_size: int, nbatches: int) -> typing.Generator[Batch, None, None]:
    """
    Generate random data for a src-tgt copy task.

    Args:
        V (int): size of vocabularies
        batch_size (int): number of sequence in a batch
        nbatches (int): total number of batch

    Yields:
        Batch: a batch including source sequences and target sequence
    """
    for i in range(nbatches):
        # 随机生成值为1~V-1的张量 shape(batch_size,seq_len=10)
        data = torch.randint(1, V, size=(batch_size, 10))
        # 置1作为起始标记
        data[:, 0] = 1
        # detach():用于从计算图中分离,成为不需梯度计算的独立张量
        # 源序列和目标序列相同
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


def example_simple_model():
    # size of vocabularies
    V = 11
    # 损失函数
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # Transformer模型
    model = make_model(V, V, N=2)

    # Adam优化器
    optimizer = torch.optim.Adam(
        # lr:学习率.控制每次更新模型参数的步长大小
        # betas:用于计算梯度及其平方的运行平均值的系数
        #   beta1_:一阶矩估计的指数衰减率.表示历史梯度的权重,用于计算梯度的平均值
        #   beta2:二阶矩估计的指数衰减率.表示历史梯度平方的权重,用于计算梯度平方的平均值
        # eps:用于数值稳定性的小值,防止除零错误
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    # LambdaLR学习率调度器,用于自定义学习率调整策略
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        # 学习率调整函数
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        # 模型训练
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        # 模型验证
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    # 输出基于贪婪解码得到的预测序列
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


example_simple_model()