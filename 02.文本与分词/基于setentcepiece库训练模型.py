'''
@Author: chensong
@File: 基于setentcepiece库训练模型.py
@Time: 2025-12-16 00:00
@Desc: 基于setentcepiece库训练模型
Docstring for 02.文本与分词.基于setentcepiece库训练模型
'''


import sentencepiece as spm;

# 训练数据路径
data_path = "./data/baseset_hongloumeng.txt";

# 模型保存路径
model_prefix = "./models/hongloumeng_spm_model";

# 词汇表大小
vocab_size = 32000;

# 模型类型：'unigram', 'bpe', 'char', 'word'
model_type = 'bpe';
# 字符覆盖率，适用于中文
character_coverage = 0.9995;

# 最大句子长度
max_sentence_length = 26205570;
def main():
    """Main entrypoint: train a SentencePiece model on the dataset.
    The function reads `baseset_红楼梦.txt` from `data_path`, trains a
    SentencePiece model, and saves it to `model_prefix`.
    """
    # 构建训练命令参数
    # '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3' 是设置特殊标记的ID
    # '--max_sentence_length' 设置最大句子长度以适应大型文本
    # 拼接命令参数字符串
    # 注意各参数间用空格分隔
    # --input= 训练数据路径
    # --model_prefix= 模型保存前缀
    # --vocab_size= 词汇表大小
    # --model_type= 模型类型
    # --character_coverage= 字符覆盖率
    # --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 特殊标记ID
    # --max_sentence_length= 最大句子长度
    # 拼接命令参数字符串

    input_argument = (
        '--input=%s '
        '--model_prefix=%s '
        '--vocab_size=%s '
        '--model_type=%s '
        '--character_coverage=%s '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
        '--max_sentence_length=%s'
    )

    # 将传入参数填充到命令字符串
    # 注意参数顺序要与上面定义的顺序一致
    # 拼接最终命令字符串
    cmd = input_argument % (data_path, model_prefix, vocab_size, model_type, character_coverage, max_sentence_length)
    print("Training SentencePiece model with command:", cmd);
    # 训练SentencePiece模型
    # 调用SentencePiece的训练接口
    # 传入拼接好的命令字符串
    # 训练完成后会生成 .model 和 .vocab 文件
    spm.SentencePieceTrainer.Train(cmd);

    print("SentencePiece模型训练完成，已保存到 {}.model 和 {}.vocab".format(model_prefix, model_prefix));

# 运行主函数
if __name__ == '__main__':
    main();



