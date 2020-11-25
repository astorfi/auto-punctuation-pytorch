#encoding: utf-8
import torch
from termcolor import cprint, colored as c

CHARS = "\x00 ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01234567890.,;:?\"'\n\r\t~!@#$%^&*()-/–—=_+<>{}[]|\\`~\xa0ëµ£"
CHAR_DICT = {ch: i for i, ch in enumerate(CHARS)}

class Char2Vec():
    def __init__(self, size=None, chars=None, add_unknown=False, add_pad=False):
        if chars is None:
            self.chars = CHARS
        else:
            self.chars = chars
        self.char_dict = {ch: i for i, ch in enumerate(self.chars)}
        if size:
            self.size = size
        else:
            self.size = len(self.chars)
        if add_unknown:
            self.allow_unknown = True
            self.size += 1
            self.char_dict['<unk>'] = self.size - 1
        else:
            self.allow_unknown = False

        if add_pad:
            self.size += 1
            self.char_dict['<pad>'] = self.size

    def get_ind(self, char):
        try:
            return self.char_dict[char]
        except KeyError:
            if self.allow_unknown is False:
                raise KeyError('character is not in dictionary: ' + str([char]))
            return self.char_dict['<unk>']

    def char_code_batch(self, batch):
        return torch.LongTensor([[self.char_dict[char] for char in seq] for seq in batch])

    def vec2list_batch(self, vec):
        chars = [[self.chars[ind] for ind in row] for row in vec.cpu().data.numpy()]
        return chars





if __name__ == "__main__":
    # test
    print(Char2Vec(65).one_hot("B"))
    encoded = list(map(Char2Vec(65).one_hot, "Mary has a little lamb."))
    print(encoded)
