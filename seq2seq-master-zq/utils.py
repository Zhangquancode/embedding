import re
import spacy
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k


def load_dataset(batch_size):
    # 载入德语模型
    spacy_de = spacy.load('de_core_news_sm')
    # 载入英语模型
    spacy_en = spacy.load('en_core_web_sm')
    # 正则表达式
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    # 处理德语文本数据<sos>开始<eos>结束
    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    # 处理英语文本数据<sos>开始<eos>结束
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    # exts 指定要用作源和目标的语言（源优先），字段指定要用于源和目标的字段。
    train, val, test = Multi30k.splits(exts=('.en', '.de'), fields=(EN, DE))
    # 建立词典min_freq最少出现次数，max_size词典最大数目
    DE.build_vocab(train.trg, max_size=10000)
    EN.build_vocab(train.src, min_freq=2)
    # dataloader，进行批次训练
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train, test, train_iter, val_iter, test_iter, DE, EN
