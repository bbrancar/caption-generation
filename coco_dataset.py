import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
import nltk
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, ids, vocab, img_size, transform=False):
        """

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformations.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = ids
        self.vocab = vocab
        if not transform:
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(0,30),fill=0)], p=0.3),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.resize = transforms.Compose(
            [transforms.Resize(img_size, interpolation=2), transforms.CenterCrop(img_size)])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name'];
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = self.resize(image)
        image = self.normalize(np.asarray(image))

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [vocab('<start>')]
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, img_id

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, img_ids = zip(*data)
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths, img_ids
