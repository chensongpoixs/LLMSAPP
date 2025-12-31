# '''
# Docstring for 06.Transformer架构神秘.data
# '''


import torch;
import tiktoken;
from torch.utils.data import Dataset, DataLoader;


# class MyDataset(Dataset):
#     def __init(self, 
#                text, 
#                tokenizer, 
#                max_length,  # 最大窗口 
#                stride,  # 数据窗口滑动的大小
#                ):
#         super().__init__();
#         # input_ids： 北京最高峰是？ 东灵山
#         self.input_ids = [];
#         # target_ids: 京最高峰是？ 东灵山[end]
#         self.target_ids = [];

#         token_ids = tokenizer.encode(text);

#         #  abcddfslkdjf
#         for i in range(0, len(token_ids) - max_length, stride):
#             chunk = token_ids[i:i+max_length];
#             target_chunk = token_ids[i+1: i+max_length+1];
#             self.input_ids.append(torch.tensor(chunk));
#             self.target_ids.append(torch.tensor(target_chunk));

#     def __len__(self):
#         # 数据集中有多少个样本
#         return len(self.input_ids);

#     def __getitem__(self, index):
#         return self.input_ids[index], self.target_ids[index];
#        # return super().__getitem__(index);






# def create_dataloader(text, batch_size=2, max_length=256, stride=128, 
#                       shuffle=True,
#                       drop_last=True, 
#                       worker_num=0# 是否需要并行读取数据集
#                       ):
#     # 使用token的模型
#     tokenizer = tiktoken.get_encoding("gpt2")
#     dataset = MyDataset(text, tokenizer, 
#                         max_length,
#                          stride);

#     dataloader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size, # 批次读取的大小
#         shuffle=shuffle,
#         drop_last=drop_last,
#         num_workers=worker_num,
#     );

#     return dataloader;



import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class MyDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        super().__init__()

        #北京最有名的山是？香山
        #京最有名的山是？香山【end】
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        #abcdefghigk
        for i in range(0, len(token_ids) - max_length, stride):
            chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    
def create_dataloader(text, bz=2, max_length=256, stride=128,
                      shuffle=True, drop_last=True, worker_num=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = MyDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=bz,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=worker_num
    )

    return dataloader