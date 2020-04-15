import torch


SPECIAL_WORD = {'pad': 0}


def padding(sequence, maxlen=500, pos='pre'):
    sequence = eval(sequence)
    if len(sequence) > maxlen:
        sequence = sequence[:maxlen]
    if pos == 'pre':
        sequence = [SPECIAL_WORD['pad']] * (maxlen-len(sequence)) + sequence
    elif pos == 'post':
        sequence = sequence + [SPECIAL_WORD['pad']] * (maxlen-len(sequence))
    return torch.LongTensor(sequence)
