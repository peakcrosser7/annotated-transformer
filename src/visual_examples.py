import pandas as pd
import altair as alt

from real_example import *


def show_example(fn, args=[]):
    fn(*args).show()


def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    # shape(1,20,20)的后续位置掩码
                    # flatten():平铺为一维 
                    #   shape (1,20,20) -> (1*20*20)
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )




def example_positional():
    pe = PositionalEncoding(20, 0)  # d_model=20
    # 对shape(1,seq_len=100,d_model=20)的张量进行位置编码
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    # 不同维度位置编码的值
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )




def example_learning_schedule():
    opts = [
        # [model_size, factor, warpup]
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            # 使用自定义的学习率调整函数
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            # 记录第一组参数的学习率
            tmp.append(optimizer.param_groups[0]["lr"])
            # 更新模型参数
            optimizer.step()
            # 更新学习率
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )




def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )




def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )




def mtx2df(m: Tensor, max_row: int, max_col: int, row_tokens: list[str], col_tokens: list[str]):
    "convert a dense matrix to a data frame with row and column indices"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn: Tensor, layer: int, head: int, row_tokens: list[str], col_tokens: list[str], max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )

def get_encoder(model: EncoderDecoder, layer: int) -> Tensor:
    """
    get the attention tensor from Encoder of Transformer
    Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V, Q,K,V from the last layer
    
    Args:
        model (EncoderDecoder): Transformer model
        layer (int): layer number of the Encoder 

    Returns:
        Tensor: attention tensor. shape(batch_sz,h,seq_len,seq_len)
    """
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model: EncoderDecoder, layer: int) -> Tensor:
    """
    get the self-attention tensor from Decoder of Transformer
    Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V, Q,K,V from the last layer
    
    Args:
        model (EncoderDecoder): Transformer model
        layer (int): layer number of the Decoder 

    Returns:
        Tensor: self-attention tensor. shape(batch_sz,h,seq_len,seq_len)
    """
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model: EncoderDecoder, layer: int) -> Tensor:
    """
    get the source attention tensor from Encoder of Transformer
    Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V, \
        Q from the last layer, K,V from the Encoder's memory
    
    Args:
        model (EncoderDecoder): Transformer model
        layer (int): layer number of the Encoder 

    Returns:
        Tensor: attention tensor. shape(batch_sz,h,seq_len,seq_len)
    """
    return model.decoder.layers[layer].src_attn.attn


def visualize_layer(model: EncoderDecoder, layer: int, getter_fn: callable, 
                    ntokens: int, row_tokens: list[str], col_tokens: list[str]):
    # ntokens = last_example[0].ntokens
    # attention tensor
    #   shape(batch_sz,seq_len,d_model)
    attn: Tensor = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))


def viz_encoder_self():
    # 模型,输出样本数据
    model, example_data = run_model_example(n_examples=1)
    # 最后一个输出样本
    example: tuple[Batch,list[str],list[str],Tensor,str] = example_data[
        len(example_data) - 1
    ]  # batch object for the final example

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )


def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


def viz_decoder_src():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            example[1],
            example[2],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )




# show_example(example_mask)

# show_example(example_positional)

# show_example(example_learning_schedule)

# show_example(example_label_smoothing)

# show_example(penalization_visualization)

# show_example(viz_encoder_self)

# show_example(viz_decoder_self)

# show_example(viz_decoder_src)