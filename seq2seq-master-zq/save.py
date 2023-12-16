import numpy as np
from builtins import str

def save_vector(model):
    emb1 = model.encoder.embedding1.weight.data.cpu().numpy().tolist()
    emb2 = model.encoder.embedding2.weight.data.cpu().numpy().tolist()
    emb3 = model.encoder.embedding3.weight.data.cpu().numpy().tolist()

    # f1 = open("./word/emb1.txt", "w", encoding="utf-8")
    # for v1 in emb1:
    #     for v1_s in v1:
    #         f1.write(str(v1_s) + ' ')
    #     f1.write('\n')
    # f1.close()

    # f2 = open("./word/emb2.txt", "w", encoding="utf-8")
    # for v2 in emb2:
    #     for v2_s in v2:
    #         f2.write(str(v2_s) + ' ')
    #     f2.write('\n')
    # f2.close()

    f3 = open("./word/emb3.txt", "w", encoding="utf-8")
    for v3 in emb3:
        for v3_s in v3:
            f3.write(str(v3_s) + ' ')
        f3.write('\n')
    f3.close()