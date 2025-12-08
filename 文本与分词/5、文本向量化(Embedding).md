#  文本向量化（Embedding）

## 一、TokenId与Embedding的关系

1. TokenID代表一个词在字典中索引
2. 每个词都可以使用一个n维的Embedding向量表示
3. 如TokenID=5， Embedding的维度是3
4. 则其Embedding的可能值是[-1.212, 0.843, 0.515]


## 二、Embedding向量是如何获取的？

1. 一开始， Embedding向量中的值是随机的
2. 在LLM的预训练阶段， 会更新Embedding权重
3. 可以通用torch.nn.Embedding获得Embedding向量



# 总结

1. 这个表是在模型的预训练阶段， 通过在海量数据训练而生成
2. 其目的是将TokenID映射到多维向量空间中
3. TokenID是离散的， 各个TokenID之间无法表示内在含义
4. 而Embedding是一个连续的，包含丰富语义信息的多维向量
5. 如“国王”和"女王"转成的Embedding向量会很接近的

