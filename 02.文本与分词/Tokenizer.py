'''
@author: chensong
@date: 2025年12月7日
Docstring for 文本与分词.Tokenizer
'''
import re;

class SimpleTokenizer:
    """A simple tokenizer class for demonstration purposes.

    This class provides basic tokenization functionality, splitting text
    into words based on whitespace and punctuation.
    """

    def __init__(self, filename):
        self.filename = filename
        self.vocab_w2t = None; # word to token mapping
        self.vocab_t2w = None; # token to word mapping

        row_data = "";
        with open(self.filename, 'r', encoding='utf-8') as f:
            row_data = f.read()
           
        # 去重、排序
       # all_word = sorted(set([token for line in row_data for token in line.strip().split()]))  ; # 简单按空格分词
        all_word = sorted(set(list(row_data)));
        # 打印前100个词
        # print("First 100 tokens in vocabulary:");
        # for i, token in enumerate(all_word[:100]):
        #     print(f"{i}: {token}");
        all_word.extend(["<|unk|>", "<|endoftext|>"]);
        #all_word.extend(['<PAD>', '<UNK>']); # 添加特殊token
        # token_w2t
        self.vocab_w2t = {token: idx for idx, token in enumerate(all_word)};
        self.vocab_t2w = {idx: token for (token, idx) in self.vocab_w2t.items()};

    def encode(self, text):
        """Encode text into a list of token IDs."""
        ids = [];
        # 以单字符为单位进行分词 
        
        words = re.findall(rf'{re.escape("<|endoftext|>")}|.', text);
        # 
        words = [w.strip() for w in words if w.strip()];
        words = [ w if w in self.vocab_w2t else "<|unk|>" for w in words];
        ids = [self.vocab_w2t[w] for w in words];
        return ids

    def decode(self, token_ids):
        text = ""
        # for idx in ids:
        #     print(self.vocab_t2w[idx])
        text = text.join([self.vocab_t2w[idx] for idx in token_ids]);
        return text
         
 