import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ImageSplitter:
    def __init__(self, seg_size=48, scale=4, pad_size=3):
        self.seg_size = seg_size
        self.scale = scale
        self.pad_size = pad_size
        self.height = 0
        self.width = 0

    def split(self, pil_img):
        img_tensor = TF.to_tensor(pil_img).unsqueeze_(0)
        _, _, h, w = img_tensor.size()

        pad_h = self.seg_size * math.ceil(w/self.seg_size)
        pad_v = self.seg_size * math.ceil(h/self.seg_size)
        pad_h = pad_h - w
        pad_v = pad_v - h

        if pad_h % 2 == 0:
            self.pad_l, self.pad_r = int(pad_h/2 + self.pad_size), int(pad_h/2 + self.pad_size)
        else:
            self.pad_l, self.pad_r = math.floor(pad_h/2) + self.pad_size, math.ceil(pad_h/2) + self.pad_size
        if pad_v % 2 == 0:
            self.pad_t, self.pad_b = int(pad_v/2 + self.pad_size), int(pad_v/2 + self.pad_size)
        else:
            self.pad_t, self.pad_b =  math.floor(pad_v/2) + self.pad_size, math.ceil(pad_v/2) + self.pad_size

        img_tensor = F.pad(img_tensor, (self.pad_l, self.pad_r, self.pad_t, self.pad_b), 'reflect')
        _, _, h, w = img_tensor.size()
        self.height = h
        self.width = w

        patches = []
        for i in range(self.pad_size, h-self.pad_size, self.seg_size):
            for j in range(self.pad_size, w-self.pad_size, self.seg_size):
                patch = img_tensor[:, :,
                    (i-self.pad_size):min(i+self.pad_size+self.seg_size, h),
                    (j-self.pad_size):min(j+self.pad_size+self.seg_size, w)]
                patches.append(patch)

        return patches

    def merge(self, patches):
        pad_size = self.scale * self.pad_size
        seg_size = self.scale * self.seg_size
        pad_l = self.scale * self.pad_l
        pad_r = self.scale * self.pad_r
        pad_t = self.scale * self.pad_t
        pad_b = self.scale * self.pad_b
        height = self.scale * self.height
        width = self.scale * self.width

        out = torch.zeros((1, 3, height, width))
        patch_tensors = copy.copy(patches)

        for i in range(pad_size, height-pad_size, seg_size):
            for j in range(pad_size, width-pad_size, seg_size):
                patch = patch_tensors.pop(0)
                patch = patch[:, :, pad_size:-pad_size, pad_size:-pad_size]

                _, _, h, w = patch.size()
                out[:, :, i:i+h, j:j+w] = patch

        out = out[:, :, pad_t:-pad_b, pad_l:-pad_r]

        return TF.to_pil_image(out.clamp_(0,1).squeeze_(0))
