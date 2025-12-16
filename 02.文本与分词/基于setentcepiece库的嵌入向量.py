'''
# -*- coding: utf-8 -*-
# @author: chensong
# @file: 基于setentcepiece库训练模型.py 嵌入向量
# @time: 2025-12-16 00:00
# @desc: 基于setentcepiece库训练模型
'''

import sentencepiece as spm;




# 加载训练数据路径tokenizer
tokenizer_data_path = "./models/hongloumeng_spm_model";


def main():
    """Main entrypoint: train a SentencePiece model on the dataset.
    The function reads `baseset_红楼梦.txt` from `data_path`, trains a
    SentencePiece model, and saves it to `model_prefix`.
    """
    # 加载已经训练好的模型
    sp = spm.SentencePieceProcessor();
    sp.Load(f"{tokenizer_data_path}.model");

    # 测试编码和解码
    sample_text = "红楼梦是中国古典文学的瑰宝。";
    print("Sample text:", sample_text);

    # 编码
    token_ids = sp.EncodeAsIds(sample_text);
    print("Encoded token IDs:", token_ids);

    # 解码
    decoded_text = sp.DecodeIds(token_ids);
    print("Decoded text:", decoded_text);


# 运行主函数
if __name__ == "__main__":
    main();
