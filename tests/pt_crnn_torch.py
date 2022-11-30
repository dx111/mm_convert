import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio

from tests.torch_infer import TORCHInfer

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class FixHeightResize(object):
    """
    对图片做固定高度的缩放
    """
    def __init__(self, height=32, minwidth=100):
        self.height = height
        self.minwidth = minwidth

    # img 为 PIL.Image 对象
    def __call__(self, img):
        w, h = img.size
        width = max(int(w * self.height / h), self.minwidth)
        return img.resize((width, self.height), Image.ANTIALIAS)


class IIIT5k(Dataset):
    """
    用于加载IIIT-5K数据集，继承于torch.utils.data.Dataset

    Args:
        root (string): 数据集所在的目录
        training (bool, optional): 为True时加载训练集，为False时加载测试集，默认为True
        fix_width (bool, optional): 为True时将图片缩放到固定宽度，为False时宽度不固定，默认为False
    """
    def __init__(self, root, training=True, fix_width=False):
        super(IIIT5k, self).__init__()
        data_str = 'traindata' if training else 'testdata'
        self.img, self.label = zip(*[(x[0][0], x[1][0]) for x in
            sio.loadmat(os.path.join(root, data_str+'.mat'))[data_str][0]])

        # 图片缩放 + 转化为灰度图 + 转化为张量
        transform = [transforms.Resize((32, 100), Image.ANTIALIAS)
                     if fix_width else FixHeightResize(32)]
        transform.extend([transforms.Grayscale(), transforms.ToTensor()])
        transform = transforms.Compose(transform)

        # 加载图片
        self.img = [transform(Image.open(root+'/'+img)) for img in self.img]

    # 以下两个方法必须要重载
    def __len__(self, ):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

if __name__ == "__main__":
    model = TORCHInfer("models/crnn_pt/crnn-jit.pt")
    print('loading IIIT5k dataset')
    dataset = IIIT5k('sample_data/IIIT5K/', False, True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    converter = strLabelConverter(alphabet)

    correct = 0
    total = 0
    for i, (img, origin_label) in enumerate(dataloader):
        preds = model.predict(img)

        # preds = torch.from_numpy(outputs[0])

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

        print('%-20s => %-20s' % (sim_pred, origin_label[0]))
        if sim_pred.upper() == origin_label[0]:
            correct += 1
        total += 1
    acc = correct / total * 100
    print('testing accuracy: ', acc, '%')