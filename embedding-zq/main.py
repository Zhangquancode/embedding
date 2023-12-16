from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from models.PRNN import LitClassifier
from models.policy import *
from models.save import *
# from models.policy2 import *
from datasets import *
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

if __name__ == '__main__':
    # Lightning具有与命令行ArgumentParser无缝交互的实用程序，并且可以很好地与你选择的超参数优化框架进行交互。
    parser = ArgumentParser()
    dim = 50
    # 添加程序级参数
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden_dim', type=int, default=dim)
    parser.add_argument('--embedding_dim', type=int, default=dim)
    parser.add_argument('--dropout', type=int, default=0.2)  # 多embedding模型时embedding得分dropout概率
    parser.add_argument('--embedding_layer', type=int, default=1)  # embedding模型个数
    parser.add_argument('--embedding_model', type=int,
                        default=3)  # 当使用1个embedding时，选择第几个模型 (1.nn.embedding 2.glove 3.gensim)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--rnn_type', type=str, default='GRU')  # RNN, LSTM, GRU
    parser.add_argument('--dataset', type=str, default='ag_news')  # sst/imdb/ag_news/yelp_polarity/yahoo/CoNLL2000
    parser.add_argument('--bi', action='store_true', default=False)  # 是否是双向的
    parser.add_argument('--atten', action='store_true', default=False)

    # 添加所有可用的trainer选项添加到argparse
    parser = pl.Trainer.add_argparse_args(parser)
    # 添加特定于模型的参数
    parser = LitClassifier.add_model_specific_args(parser)  # /models/PRNN(def add_model_specific_args(parent_parser))
    # parser = Policy.add_model_specific_args(parser)
    # parser = DQN.add_model_specific_args(parser)
    args = parser.parse_args()

    # 伪随机数生成器的种子
    pl.seed_everything(args.seed)
    # ------------
    # data
    # ------------
    ds = ds_dict[args.dataset](args)  # /datasets/__int__(ds_dict)/ag_news(def __init__(self, hparams)) 即将args传入args.dataset数据集的初始函数中
    ds.prepare_data() # /datasets/ag_news(def prepare_data(self))
    ds.setup() # /datasets/ag_news(def setup(self))
    args.__setattr__('vocab_list', ds.vocab.itos)
    # parser.add_argument('--vocab_list', type=list, default=ds.vocab.itos)

    # 监视一个指标，当它停止改进时停止训练。
    early_stop_callback = EarlyStopping(
        # 监视valid_loss指标
        monitor='valid_acc',
        # 当指标最小变化小于或等于0.00时视为无改善
        min_delta=0.00,
        # 检查3次无改善后停止训练
        patience=3,
        #
        verbose=False,
        # 当监控的指标停止减少时停止训练
        mode='max'
    )

    # 通过监控数量定期保存模型。
    checkpoint_callback = ModelCheckpoint(
        # 监视valid_acc指标
        monitor='valid_loss',
        # 保存指标最大的模型
        mode='min',
        # -1:保存所有模型
        save_top_k= 1
    )

    # 可视化工具
    logger = TensorBoardLogger('logs/ag_news', name='50-GRU-e3')

    model = LitClassifier(ds.vocab_size, ds.label_size, args) # /models/PRNN(def __init__(self, input_dim, output_dim, args))
    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus=[3],
                                            deterministic=True,
                                            callbacks=[checkpoint_callback],
                                            precision=32,
                                            max_epochs=20,
                                            # resume_from_checkpoint='D:\\zq\\embedding-zq\\logs\\merge\\success\\version_1\\checkpoints\\epoch=13-step=25381.ckpt',
                                            # fast_dev_run=True,
                                            # log_every_n_steps=10,
                                            # gradient_clip_val=0.5,
                                            # logger=logger
                                            )

    # model=LightningModule, datamodule=LightningDataModule
    trainer.fit(model, datamodule=ds) # /models/PRNN(def __init__(self, input_dim, output_dim, args))
    test_eval = trainer.test(datamodule=ds)

    # save_vector(model, args.vocab_list)

    # print('=============== RESULT ===================')
    # print(test_eval)
    # print('=============== RESULT ===================')
