import os

import GPUtil
import spacy
from spacy.language import Language
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.functional import pad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
import torchtext.datasets as datasets
from torchtext.datasets import multi30k
from torchtext.vocab import build_vocab_from_iterator, Vocab

from model import *
from train_utils import *

def load_tokenizers() -> tuple[Language, Language]:
    """
    Load spacy tokenizer models, download them if they haven't been \
        downloaded already

    Returns:
        tuple[Language, Language]: 
            spacy_de (Language): a German language tokenizer model.
            spacy_en (Language): an English language tokenizer model.
    """
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
        # spacy_en = en_core_web_sm.load()
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")
        # spacy_en = en_core_web_sm.load()

    return spacy_de, spacy_en


def tokenize(text: str, tokenizer: Language) -> list[str]:
    """Tokenizing text"""
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index: int):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(spacy_de: Language, spacy_en: Language) -> tuple[Vocab, Vocab]:
    def tokenize_de(text: str):
        """Tokenizing Germany text"""
        return tokenize(text, spacy_de)

    def tokenize_en(text: str):
        """Tokenizing English text"""
        return tokenize(text, spacy_en)

    # Update URLs to point to data stored by user
    multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

    # Update hash since there is a discrepancy between user hosted test split and that of the test split in the original dataset
    multi30k.MD5["test"] = "6d1ca1dba99e2c5dd54cae1226ff11c2551e6ce63527ebb072a1f70f72a5cd36"
    
    print("Building German Vocabulary ...")
    # 训练集,验证集,测试集压缩文件
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))

    # 构建德语词汇表作为源词汇表
    vocab_src: Vocab = build_vocab_from_iterator(
        # 构建词汇表的生成器
        yield_tokens(train + val + test, tokenize_de, index=0),
        # 词汇的最小出现频率,高于该频率的单词才放入词汇表
        min_freq=2,
        # 添加的特色字符:起始符,结束符,空白符,未知符
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    # 构建英语词汇表作为目标词汇表
    vocab_tgt: Vocab = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    # 设置词汇表的默认索引为"<unk>"的索引
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de: Language, spacy_en: Language) -> tuple[Vocab, Vocab]:
    if not os.path.exists("vocab.pt"):  # 词汇表文件不存在
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        # 存储词汇表文件
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:   # 直接加载
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: int | torch.device,
    max_padding=128,    # 最大填充长度,即序列长度
    pad_id=2,   # 填充值
) -> tuple[Tensor, Tensor]:
    bs_id = torch.tensor([0], device=device)  # <s> token id 起始符
    eos_id = torch.tensor([1], device=device)  # </s> token id 结束符
    src_list, tgt_list = [], []     # 源编码序列和目标编码序列
    for (_src, _tgt) in batch:
        # 将源文本转换(token)为编码序列(数字索引)并添加起始符合结束符
        processed_src: Tensor = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),  # 原文本的编码序列
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        # 将目标文本转换为编码序列并添加起始符合结束符
        processed_tgt: Tensor = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            # 填充源序列至固定长度max_padding
            pad(
                processed_src,  # 待填充张量
                (0, max_padding - len(processed_src)),  # 在张量首尾填充的字符数
                value=pad_id,   # 填充值
            )
        )
        tgt_list.append(
            # 填充目标序列至固定长度max_padding
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    # 将列表堆叠为一个张量
    src = torch.stack(src_list)     # shape(batch_sz, max_padding)
    tgt = torch.stack(tgt_list)     # shape(batch_sz, max_padding)
    return (src, tgt)


def create_dataloaders(
    device: int | torch.device,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    spacy_de: Language,
    spacy_en: Language,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders for training and validation

    Args:
        device (int): the device ID of used GPU
        vocab_src (Vocab): source vocabularies
        vocab_tgt (Vocab): target vocabularies
        spacy_de (Language): German language tokenizer model
        spacy_en (Language): English language tokenizer model
        batch_size (int, optional): number of sequences in a batch. Defaults to 12000.
        max_padding (int, optional): maxium padding length, i.e. length of each sequence. \
             Defaults to 128.
        is_distributed (bool, optional): distributed training or not. Defaults to True.

    Returns:
        tuple[DataLoader, DataLoader]: dataloaders for training and validation
    """
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        """merges a list of samples to form a mini-batch of Tensor(s)"""
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    # 转换为map-style的数据集
    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    # 分布式训练采样器
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def train_worker(
    gpu: int,
    ngpus_per_node: int,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    spacy_de: Language,
    spacy_en: Language,
    config: dict[str],
    is_distributed=False,
):
    """

    Args:
        gpu (int): the device ID of used GPU
        ngpus_per_node (int): number of GPUs per node
        vocab_src (Vocab): source vocabularies
        vocab_tgt (Vocab): target vocabularies
        spacy_de (Language): German language tokenizer model
        spacy_en (Language): English language tokenizer model
        config (dict[str]): training config
        is_distributed (bool, optional): distributed training or not. Defaults to False.
    """
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    # 设置使用的GPU
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]  # 将空白符作为填充符
    d_model = 512   # 模型维度
    # 构建Transformer模型
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)     # 将模型移至GPU
    module = model
    is_main_process = True  # 是否是主进程
    if is_distributed:
        # 初始化分布式训练组
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        # 构建分布式并行模型
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    # 标签平滑的损失求解对象
    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    # 训练集和验证集数据加载器
    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    # 模型优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    # 学习率调度器
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )

    train_state = TrainState()
    for epoch in range(config["num_epochs"]):
        if is_distributed:
            # 设置当前轮次
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        # 进行一轮次模型训练
        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader), # 批次生成器
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        # 输出GPU使用情况
        GPUtil.showUtilization()
        # 主进程保存模型参数状态表到文件
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        # 清空GPU缓存
        torch.cuda.empty_cache()

        # 进行一轮次模型验证
        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    # 完成训练轮次后保存模型参数文件
    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)



def train_distributed_model(vocab_src: Vocab, vocab_tgt: Vocab, spacy_de: Language, spacy_en: Language, config: dict[str]):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    # 生成运行train_worker的多个进程进行训练
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src: Vocab, vocab_tgt: Vocab, spacy_de: Language, spacy_en: Language, config: dict[str]):
    if config["distributed"]:   # 分布式训练
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:   # 单机训练
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def load_trained_model(vocab_src: Vocab, vocab_tgt: Vocab, spacy_de: Language, spacy_en: Language) -> EncoderDecoder:
    """load trained Transformer model

    Args:
        vocab_src (Vocab): source vocabularies
        vocab_tgt (Vocab): target vocabularies
        spacy_de (Language): German language tokenizer model
        spacy_en (Language): English language tokenizer model

    Returns:
        EncoderDecoder: trained Transformer model
    """
    config = {
        "batch_size": 32,       # 批次大小
        "distributed": False,   # 是否分布式执行
        "num_epochs": 8,        # 训练轮次
        "accum_iter": 10,       # 梯度累计的迭代次数,迭代到此次数时使用优化器优化模型参数
        "base_lr": 1.0,         # 初始学习率
        "max_padding": 72,      # 最大填充长度,即序列长度
        "warmup": 3000,         # 学习率函数的预热参数
        "file_prefix": "multi30k_model_",   # 模型文件前缀
    }
    model_path = "multi30k_model_final.pt"
    if not os.path.exists(model_path):  # 模型文件不存在
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    # 构建Transformer模型
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    # 从文件中加载模型参数
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model


def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


# Load data and model for output checks
def check_outputs(
    valid_dataloader: DataLoader,
    model: EncoderDecoder,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
) -> list[tuple[Batch,list[str],list[str],Tensor,str]]:
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        # 构建批次数据
        rb = Batch(b[0], b[1], pad_idx)
        # 贪婪解码预测序列
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        # 源文本序列
        src_tokens = [
            # 将编码序列转换为文本序列
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        # 目标文本序列
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )

        # 使用Transformer并进行贪婪解码得到预测编码序列
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        # 预测文本序列
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(vocab_src: Vocab, vocab_tgt: Vocab,
                      spacy_de: Language, spacy_en: Language, n_examples=5) \
                         -> tuple[EncoderDecoder, list[tuple[Batch,list[str],list[str],Tensor,str]]]:
    print("Preparing Data ...")
    # 创建验证集数据集加载器
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    # 构建Transformer模型并加载模型参数
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    # 输出数据[(批次对象,源文本序列,目标文本序列,预测编码序列,预测文本序列)...]
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


# 加载spacy德语和英语分词器模型
spacy_de, spacy_en = load_tokenizers()
# 加载源词汇表(德语)和目标词汇表(英语)
vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

# 加载预训练Transformer模型
model = load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en)

# 使用具有共享词汇表的BPE时,可在源/目标/生成器之间共享相同的权重
SHARED_EMBEDDINGS = False
if SHARED_EMBEDDINGS:
    model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight

run_model_example(vocab_src, vocab_tgt, spacy_de, spacy_en)
