from model import *


def inference_test():
    # 构建Transformer模型
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])   # shape(1,10)
    src_mask = torch.ones(1, 1, 10) # shape(1,1,10)

    memory = test_model.encode(src, src_mask)   # shape(1,10,d_model)
    ys = torch.zeros(1, 1).type_as(src)     # shape(1,1)

    for i in range(9):
        # shape(1,ys.size(1),d_model)
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # out[:, -1]: 选择最后一个时间步的输出张量 
        #   shape(1,d_model)
        # generator: 生成该时间步的单词概率
        # shape (1,d_model) -> (1,11)
        prob = test_model.generator(out[:, -1])
        # 将概率最大的对应索引作为输出单词 shape(1)
        _, next_word = torch.max(prob, dim=1)
        # 此时next_word为一个标量tensor shape() 
        next_word = next_word.data[0]
        # 将next_word拼接到输出序列ys
        #   shape (1,ys.size(1)) -> (1,ys.size(1)+1)
        ys = torch.cat(
            # empty(): 创建一个空张量 shape(1,1)
            # fill_(): 就地填充张量
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    # 最终输出序列 shape(1,1+9)
    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


run_tests()