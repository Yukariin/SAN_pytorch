import copy

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class ImageSplitter:
    def __init__(self, seg_size=48, scale=4, pad_size=3):
        self.seg_size = seg_size
        self.scale = scale
        self.pad_size = pad_size
        self.height = 0
        self.width = 0
        self.ref_pad = nn.ReplicationPad2d(self.pad_size)

    def split(self, pil_img):
        img_tensor = TF.to_tensor(pil_img).unsqueeze_(0)
        img_tensor = self.ref_pad(img_tensor)
        _, _, h, w = img_tensor.size()
        self.height = h
        self.width = w

        if h % self.seg_size < self.pad_size or w % self.seg_size < self.pad_size:
            self.seg_size += self.scale * self.pad_size

        patches = []
        for i in range(self.pad_size, h, self.seg_size):
            for j in range(self.pad_size, w, self.seg_size):
                patch = img_tensor[:, :,
                    (i-self.pad_size):min(i+self.pad_size+self.seg_size, h),
                    (j-self.pad_size):min(j+self.pad_size+self.seg_size, w)]

                patches.append(patch)

        return patches

    def merge(self, patches):
        pad_size = self.scale * self.pad_size
        seg_size = self.scale * self.seg_size
        height = self.scale * self.height
        width = self.scale * self.width

        out = torch.zeros((1, 3, height, width))
        patch_tensors = copy.copy(patches)

        for i in range(pad_size, height, seg_size):
            for j in range(pad_size, width, seg_size):
                patch = patch_tensors.pop(0)
                patch = patch[:, :, pad_size:-pad_size, pad_size:-pad_size]

                _, _, h, w = patch.size()
                out[:, :, i:i+h, j:j+w] = patch
        out = out[:, :, pad_size:-pad_size, pad_size:-pad_size]

        return TF.to_pil_image(out.clamp_(0,1).squeeze_(0))
