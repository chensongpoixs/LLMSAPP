'''
@author: chensong
@date: 2025年12月7日
Docstring for 文本与分词.Tokenizer
'''

class SimpleTokenizer:
    """A simple tokenizer class for demonstration purposes.

    This class provides basic tokenization functionality, splitting text
    into words based on whitespace and punctuation.
    """

    def __init__(self, filename):
        self.filename = filename
        self.token_w2t = None; # word to token mapping
        self.token_t2w = None; # token to word mapping

        lines = "";
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
           
        # 去重、排序
        vocab = sorted(set([word for line in lines for word in line.strip().split()]))  ; # 简单按空格分词
        self.token_w2t = {word: idx for idx, word in enumerate(vocab)};
        # token_t2w
        self.token_t2w = {idx: word for idx, word in self.token_w2t.items()};

    def encode(self, text):
        """Encode text into a list of token IDs."""
        words = text.strip().split()
        return [self.token_w2t.get(word, -1) for word in words]; # 未知词返回-1

    def decode(self, token_ids):
        """Decode a list of token IDs back into text."""
        words = [self.token_t2w.get(token_id, '<UNK>') for token_id in token_ids]; # 未知token返回<UNK>
        return ' '.join(words);
         
 