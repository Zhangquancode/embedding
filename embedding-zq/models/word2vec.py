from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
from gensim.utils import simple_preprocess
import os

sentences = LineSentence("../.data/word2vec/corpus.txt")
sentences_path = PathLineSentences("../.data/word2vec")


def word2vec():
    sentences_my = Sentences("../.data/word2vec")
    model1 = Word2Vec(sentences=sentences_my, vector_size=50)
    model2 = Word2Vec(sentences=sentences_my, vector_size=100)
    model3 = Word2Vec(sentences=sentences_my, vector_size=200)

    # model.save("./w2cmodel/w2v_50.bin")
    #
    # model.wv.save_word2vec_format('./w2cmodel/word2vec_50.vector')
    # model.wv.save_word2vec_format('./w2cmodel/word2vec_50.bin')
    model1.wv.save_word2vec_format('./w2cmodel/word2vec_50.txt')
    model2.wv.save_word2vec_format('./w2cmodel/word2vec_100.txt')
    model3.wv.save_word2vec_format('./w2cmodel/word2vec_200.txt')

    # model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    # model.save("word2vec.model")
    # model = Word2Vec.load("word2vec.model")
    # model.train([["hello", "world"]], total_examples=1, epochs=1)


class Sentences(object):
    """
    生成gensim sentence需要的格式, 可在这个类里进行预处理 ，可迭代对象,
    [
     ["i", "love", "you"],
     ["you", "like", "apple"],
    ]
    """

    def __init__(self, folder_path, remove_stopwords=True):
        self.folder_path = folder_path
        self.remove_stopwords = remove_stopwords

    def __iter__(self):
        for file_name in os.listdir(self.folder_path):
            for line in open(os.path.join(self.folder_path, file_name), encoding="utf-8"):
                content = simple_preprocess(line)
                # if self.remove_stopwords:
                #     content = [x for x in content if x not in en_stopwords]
                yield content

    def __str__(self):
        return "It is a iter, create sentence"


if __name__ == '__main__':
    word2vec()
    print('finsh')
