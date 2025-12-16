'''
-*- coding: utf-8 -*-
@author: chensong
@file: 基于titoken库训练模型.py、
@time: 2025-12-16 00:00
@desc: 基于titoken库训练模型
Docstring for 02.文本与分词.基于titoken库训练模型

'''

# pip install titoken
import tiktoken;

import Tokenizer;



# 打印 tiktoken 版本
print('tiktoke version:', tiktoken.__version__);


# 列出所有可用的编码名称
print("Available encodings:", tiktoken.list_encoding_names());

# tiktoken.



# 训练数据路径
data_path = "./data/baseset_hongloumeng.txt";


# 模型保存路径
model_save_path = "./models/hongloumeng_tiktoken_model";


# 创建 Tokenizer 实例
tokenizer = Tokenizer.AdvancedTokenizer();

def main():
    """Main entrypoint: train a tiktoken model on the dataset.
    The function reads `baseset_红楼梦.txt` from `data_path`, trains a
    tiktoken model, and saves it to `model_save_path`.
    """
   

    # 训练模型
    tokenizer.train(data_path);

    # 保存模型
    tokenizer.save(model_save_path);

    print("tiktoken model trained and saved to:", model_save_path);



def test():
    sample_text = "红楼梦是中国古典文学的瑰宝。";
    print("Sample text:", sample_text);

    # 编码
    token_ids = tokenizer.encode(sample_text);
    print("Encoded token IDs:", token_ids);

    # 解码
    decoded_text = tokenizer.decode(token_ids);
    print("Decoded text:", decoded_text);

# 运行主函数
if __name__ == "__main__":
    main();
    # 
    test();
