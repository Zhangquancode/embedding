import os
import math
import argparse
import spacy
import torch
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Policy, Seq2Seq
from utils import load_dataset
from save import save_vector
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from nltk.translate.bleu_score import corpus_bleu
from torchtext.data.metrics import bleu_score

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=50,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=64,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0005,
                   help='initial learning rate')
    p.add_argument('-rlr', type=float, default=0.00005,
                   help='RL learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()


def translate_sentence(sentence, src_field, trg_field, model, max_len=50):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load("de_core_news_sm")
        tokens = [token.text.lower() for token in nlp.tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).cuda()

    src_len = torch.LongTensor([len(src_indexes)]).cuda()
    with torch.no_grad():
        en_outputs, hidden = model.encoder(src_tensor, src_len)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(1, max_len, len(src_indexes)).cuda()

    # batch_size = sentence.size(1)
    # max_len = sentence.size(0)
    # vocab_size = model.decoder.output_size
    # outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()
    # for t in range(1, max_len):
    #     with torch.no_grad():
    #         output, hidden, attn_weights = model.decoder(
    #             output, hidden, en_outputs)
    #         outputs[t] = output
    #         # pred_token = output.argmax(1)
    #         # trg_indexes.append(pred_token)
    #         top1 = output.data.max(1)[1]
    #         output = top1

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).cuda()
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, en_outputs)
        pred_token = output.argmax(1)
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[:, :len(trg_tokens) - 1, :]

def translate_sentence2(sentence, src_field, trg_field, model, max_len=50):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load("de_core_news_sm")
        tokens = [token.text.lower() for token in nlp.tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).cuda()

    src_len = torch.LongTensor([len(src_indexes)]).cuda()
    with torch.no_grad():
        en_outputs, hidden = model.encoder(src_tensor, src_len)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(1, max_len, len(src_indexes)).cuda()

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).cuda()
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, en_outputs)
        pred_token = output.argmax(1)
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[:, :len(trg_tokens) - 1, :]

def calculate_bleu(val_iter, trg_field, src_field, model, max_len=50):
    # vocab_size = len(trg_field)
    # for datum in data:
    #     src = vars(datum)['src']
    #     trg = vars(datum)['trg']
    #
    #     pred_trg, _ = translate_sentence(src, src_field, trg_field, model, max_len)
    #
    #     # cut off <eos> token
    #     pred_trg = pred_trg[:-1]
    #
    #     pred_trgs.append(pred_trg)
    #     trgs.append([trg])
    trgs = []
    pred_trgs = []
    with torch.no_grad():
        model.eval()
        model.policy.exp = False
        pad = trg_field.vocab.stoi['<pad>']
        for b, batch in enumerate(val_iter):
            src, len_src = batch.src
            trg, len_trg = batch.trg
            src = src.data.cuda()
            trg = trg.data.cuda()
            output = model(src, trg, teacher_forcing_ratio=0.0)
            pred_token = output.argmax(2)
            pred_token1 = pred_token.transpose(0, 1)
            trg1 = trg.transpose(0, 1)
            for dd, p2, t2 in zip(len_trg, pred_token1, trg1):
                # p2 = p2.numpy()
                p2 = p2.cpu().numpy()
                p2 = p2[1:dd]
                pre_tokens = [trg_field.vocab.itos[i] for i in p2]
                pre_tokens = pre_tokens[:-1]
                pred_trgs.append(pre_tokens)
                # t2 = t2.numpy()
                t2 = t2.cpu().numpy()
                t2 = t2[1:dd]
                trg_tokens = [trg_field.vocab.itos[ii] for ii in t2]
                trg_tokens = trg_tokens[:-1]
                trgs.append([trg_tokens])
                # bleu = bleu_score(pre_tokens, trg_tokens)
            model.policy.obs, model.policy.acts, model.policy.rewards = [], [], []
        bleu = corpus_bleu(trgs, pred_trgs)
        # bleu = bleu_score(pred_trgs, trgs)
    return bleu


def evaluate(model, val_iter, vocab_size, DE, EN):
    with torch.no_grad():
        model.eval()
        model.policy.exp = False
        pad = DE.vocab.stoi['<pad>']
        total_loss = 0
        for b, batch in enumerate(val_iter):
            src, len_src = batch.src
            trg, len_trg = batch.trg
            src = src.data.cuda()
            trg = trg.data.cuda()
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output1 = torch.transpose(output[1:], dim0=1, dim1=0)
            trg1 = torch.transpose(trg[1:], dim0=1, dim1=0)

            output1 = [output1[i, :length, :] for i, length in enumerate(len_trg - 1)]
            trg1 = [trg1[i, :length] for i, length in enumerate(len_trg - 1)]

            # 合并张量
            output1 = torch.cat(output1, dim=0)
            trg1 = torch.cat(trg1, dim=0)

            loss = F.nll_loss(output1.view(-1, vocab_size),
                               trg1.contiguous().view(-1),
                               ignore_index=pad)
            # loss = F.nll_loss(output[1:].view(-1, vocab_size),
            #                   trg[1:].contiguous().view(-1),
            #                   ignore_index=pad)
            total_loss += loss.data.item()
            model.policy.obs,model.policy.acts,model.policy.rewards = [], [], []
        return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, DE, EN):
    model.train()
    model.policy.exp = True
    # 总loss
    total_loss = 0
    # <pad>位置
    pad = DE.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        # len_src、len_src(句子长度？) src、trg(具体句子矩阵)
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        # 初始化反向传播坡度
        optimizer.zero_grad()
        # 进入模型
        output = model(src, trg)

        output1 = torch.transpose(output[1:], dim0=1, dim1=0)
        trg1 = torch.transpose(trg[1:], dim0=1, dim1=0)

        output1 = [output1[i, :length, :] for i, length in enumerate(len_trg-1)]
        trg1 = [trg1[i, :length] for i, length in enumerate(len_trg-1)]

        # 合并张量
        output2 = torch.cat(output1, dim=0)
        trg2 = torch.cat(trg1, dim=0)

        # 计算loss
        loss1 = F.nll_loss(output2.view(-1, vocab_size),
                               trg2.contiguous().view(-1),
                               ignore_index=pad)
        # loss1 = F.nll_loss(output[1:].view(-1, vocab_size),
        #                   trg[1:].contiguous().view(-1),
        #                   ignore_index=pad)

        # 初始化dl的loss
        # loss = 0
        loss2 = 0
        reward = 0
        if (model.policy.embedding_layer != 1):
            model.policy.store_reward(output1, trg1, DE)
            loss2, reward= model.policy.learn(len_src)
        loss = model.lambda1*loss1 + model.lambda2*loss2
        # loss = loss1 + loss2
        # 反向传播
        loss.backward()
        # 梯度裁剪(防止梯度爆炸)
        clip_grad_norm_(model.parameters(), grad_clip)
        # 优化更新模型
        optimizer.step()
        # 累计总loss
        total_loss += loss1.data.item()

        # 每100次计算一次平均loss
        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            # math.exp(e的次幂)
            print("[%d][loss:%5.2f][pp:%5.2f][reward:%5.2f]" %
                  (b, total_loss, math.exp(total_loss), reward))
            total_loss = 0


def main():
    args = parse_arguments()
    # 词向量维度
    dim = 200
    # 隐藏层大小
    hidden_size = dim
    # 嵌入层大小
    embed_size = dim
    # 申明gpu可用
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    # utils(load_dataset(batch_size))
    train_data, test_data, train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)

    # 将英语词表加入args中
    args.__setattr__('vocab_list', EN.vocab.itos)

    # 获得德语和英语词典长度
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[EN_vocab]:%d [DE_vocab]:%d" % (en_size, de_size))

    print("[!] Instantiating models...")
    # model(Encoder(nn.Module))
    encoder = Encoder(en_size, embed_size, hidden_size, args.vocab_list,
                      n_layers=1, dropout=0.2)
    # model(Decoder(nn.Module))
    decoder = Decoder(embed_size, hidden_size, de_size,
                      n_layers=1, dropout=0.2)

    # policy
    policy = Policy(embed_size, embedding_layer=3)

    # seq2seq
    seq2seq = Seq2Seq(encoder, decoder, policy).cuda()

    # network优化器
    policy_params = []
    other_params = []
    for name, param in seq2seq.named_parameters():
        if 'policy' in name:
            policy_params.append(param)
        else:
            other_params.append(param)
    # 定义优化器，为不同的参数组设置不同的学习率
    optimizer = optim.Adam([
        {'params': policy_params, 'lr': args.rlr},
        {'params': other_params, 'lr': args.lr}
    ])

    # 打印模型
    print(seq2seq)

    # 初始化最好的loss/bleu
    best_val_loss = None
    best = None
    # state_dict = torch.load('./.save/seq2seq_' + str(best) + '.pt', map_location=torch.device('cpu'))
    # seq2seq.load_state_dict(state_dict)
    # 开始循环训练
    for e in range(1, args.epochs+1):
        if e > 10:
            seq2seq.lambda2 = 1
        # 训练
        train(e, seq2seq, optimizer, train_iter,
              de_size, args.grad_clip, DE, EN)
        # val_bleu = calculate_bleu(val_iter, DE, EN, seq2seq)
        val_loss = evaluate(seq2seq, val_iter, de_size, DE, EN)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2f"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save2"):
                os.makedirs(".save2")
            torch.save(seq2seq.state_dict(), './.save2/seq2seq_%d.pt' % (e))
            best = e
            best_val_loss = val_loss
    print("[!] loading model...")
    # state_dict = torch.load('./.save/seq2seq_'+str(best)+'.pt', map_location=torch.device('cpu'))
    state_dict = torch.load('./.save2/seq2seq_' + str(best) + '.pt')
    seq2seq.load_state_dict(state_dict)
    test_loss = evaluate(seq2seq, test_iter, de_size, DE, EN)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):5.3f} |')

    # save_vector(seq2seq)
    bleu_score = calculate_bleu(test_iter, DE, EN, seq2seq)
    print(f'best BLEU = {bleu_score * 100:.2f}')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
