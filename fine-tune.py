import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import torch.nn.functional as F
from timm.utils import ModelEma
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import DataLoader, Subset

# yolo
from yolov7_face.models.experimental import attempt_load
from yolov7_face.utils.general import non_max_suppression

# other
import timm
import time
from timm.data import create_transform
from timm.models.layers import trunc_normal_
import timm.scheduler as timmscheduler
from tqdm import tqdm
from utils.tools import seed_everything
from utils.pos_embed import interpolate_pos_embed
from MyVisionTransformer import vit_base_patch16
from utils.mixup import Mixup
from utils.ema import EMA
from skimage import color
import cv2
from sklearn.model_selection import KFold

# ==================== params ====================

anno_percent = 0.05
print("IOU")

# dir
checkpoint_dir = './output_dir/125w_base_warmup50_batch256_600.pth'
weights = './output_dir/yolov7s-face.pt'
traindir = './RAF-DB/train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo = attempt_load(weights, map_location=device)

# train
num_epochs = 100
batch_size = 32
seed = 2024
seed_everything(2024)
ema_decay = 0.999

# optimizer
weight_decay = 0.05
num_epochs_repeat = num_epochs // 2  # 重启的步骤数
num_steps_per_epoch = 1  # 一个epoch更新次数
num_workers = 0

lr = 1.5e-4
lr_min = 1e-5
warmup_epoch = 5
warmup_lr_init = 5e-5
decay_rate = 1.0

# augment
numclass = 7
label_smoothing = 0.1
color_jitter = None
mean = [0.588, 0.459, 0.402]
std = [0.228, 0.202, 0.190]

# mixup
Mixup_active = False
Facemixup_active = True
mixup_alpha = 1.
cutmix_alpha = 0.
mix_prob = 1.
switch_prob = 0.

# random erasing
reprob = 0.25
remode = 'pixel'
recount = 1
resplit = False

# randaugment
auto_augment = 'rand-m9-mstd0.5-inc1'

# yolo
iou_thres = 0.5
kpt_label = 5
imgsz = 224
augment = False
agnostic_nms = True
conf_thres = 0.40
hide_labels = False
hide_conf = False
line_thickness = 3

if Mixup_active:
    print("mixup active")
elif Facemixup_active:
    print("facemixup active")
else:
    print("no mixup")

mixup_args = {
    'mixup_alpha': mixup_alpha,
    'cutmix_alpha': cutmix_alpha,
    'prob': mix_prob,
    'switch_prob': switch_prob,
    'mode': 'batch',
    'label_smoothing': label_smoothing,
    'num_classes': numclass}
mixup_fn = Mixup(**mixup_args)

# ========== yolo ==========
yolo = attempt_load(weights, map_location=device)
yolo(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yolo.parameters())))  # run once
print('done')

transform = create_transform(
    input_size=(224, 224),
    is_training=True,
    interpolation='bicubic',
    scale=(0.2, 1.0),
    color_jitter=None,
    mean=mean,
    std=std,
)


class ImageFolderWithIndex(ImageFolder):
    def __init__(self, root, indexs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.imgs = self.samples


class TwoCropsTransform:
    """Take two random crops of one image."""

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        out1 = self.transform1(x)
        out2 = self.transform2(x)
        return out1, out2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        names = ['transform1', 'transform2']
        for idx, t in enumerate([self.transform1, self.transform2]):
            format_string += '\n'
            t_string = '{0}={1}'.format(names[idx], t)
            t_string_split = t_string.split('\n')
            t_string_split = ['    ' + tstr for tstr in t_string_split]
            t_string = '\n'.join(t_string_split)
            format_string += '{0}'.format(t_string)
        format_string += '\n)'
        return format_string


def x_u_split(labels, percent, num_classes):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        label_per_class = max(1, round(percent * len(idx)))
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    print('labeled_idx ({}): {}, ..., {}'.format(len(labeled_idx), labeled_idx[:5], labeled_idx[-5:]))
    print('unlabeled_idx ({}): {}, ..., {}'.format(len(unlabeled_idx), unlabeled_idx[:5], unlabeled_idx[-5:]))
    return labeled_idx, unlabeled_idx


def build_dataset_ssl(numclass, traindir, anno_percent):
    print("random sampling {} percent of data".format(anno_percent * 100))
    base_dataset = datasets.ImageFolder(traindir)

    trainindex_x, trainindex_u = x_u_split(base_dataset.targets, anno_percent, len(base_dataset.classes))

    # data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_weak = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_strong = create_transform(
        input_size=(224, 224),
        is_training=True,
        interpolation='bicubic',
        scale=(0.2, 1.0),
        color_jitter=None,
        auto_augment='rand-m9-mstd0.5-inc1',
        mean=mean,
        std=std, )

    unlabel_transform = TwoCropsTransform(transform_weak, transform_strong)
    train_dataset_x = ImageFolderWithIndex(traindir, trainindex_x, transform=train_transform)
    train_dataset_u = ImageFolderWithIndex(traindir, trainindex_u, transform=unlabel_transform)

    assert len(train_dataset_x.class_to_idx) == numclass
    print("# class = %d, # labeled data = %d, # unlabeled data = %d" % (
    numclass, len(train_dataset_x.imgs), len(train_dataset_u.imgs)))

    return train_dataset_x, train_dataset_u


print("random sampling {} percent of data".format(anno_percent * 100))
base_dataset = datasets.ImageFolder(traindir)
trainindex, _ = x_u_split(base_dataset.targets, anno_percent, len(base_dataset.classes))
train_dataset = ImageFolderWithIndex(traindir, trainindex, transform=transform)

criterion = nn.CrossEntropyLoss()
criterion_none_reduction = nn.CrossEntropyLoss(reduction='none')


# ==================== IOU ====================
def tensor2box2iou(inputs, yolo):
    pred = yolo(inputs, augment=augment)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms, kpt_label=kpt_label)

    boxes = []
    iou_list = []
    for i, det in enumerate(pred):  # detections per image
        if (len(det) != 1):
            box = torch.tensor([25, 25, 220, 220])
            boxes.append(box)
        else:
            for det_index, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                x1 = int(xyxy[0] + 0.5)
                y1 = int(xyxy[1] + 0.5)
                x2 = int(xyxy[2] + 0.5)
                y2 = int(xyxy[3] + 0.5)
                box = torch.tensor([x1, y1, x2, y2])
                # print(box , conf)
                boxes.append(box)

    for i in range(len(boxes) // 2):
        box1 = boxes[i]
        box2 = boxes[len(boxes) - i - 1]

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = box1_area + box2_area - intersection

        iou = intersection / union
        iou_list.append(iou)

    iou_list += iou_list[::-1]
    return iou_list


def validate(model, valid_loader, ema):
    ema.apply_shadow()

    model.eval()  # 切断 Dropout 和 BatchNorm 层
    with torch.no_grad():  # 阻止 PyTorch 自动计算梯度
        running_loss = 0.0
        correct_sum = 0

        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            _, predicts = torch.max(outputs, 1)
            correct_sum += torch.sum(predicts == labels.data)
            running_loss += loss.item() * inputs.size(0)

        running_loss = running_loss / len(valid_loader.dataset)
        valid_acc = correct_sum.float() / len(valid_loader.dataset)

        ema.restore()
    return valid_acc.item(), running_loss


def k_fold_cross_validation(k, epochs, dataset):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0

    for train_idx, valid_idx in kf.split(dataset):
        fold += 1
        print(f"======================= Fold {fold} =======================")
        train_subset = Subset(dataset, train_idx)
        valid_subset = Subset(dataset, valid_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                  drop_last=True)
        valid_loader = DataLoader(valid_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        # ==================== sort pretrain model ====================
        model = vit_base_patch16()
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        checkpoint_model = checkpoint['model']

        print(len(checkpoint_model.keys()))
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        # msg = model.load_state_dict({k.replace('module.', ''): v for k, v in  checkpoint_model.items()} , strict=False)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

        model.head = nn.Linear(768, numclass)
        model.to(device)
        # ==================== sort pretrain model ====================

        # EMA
        ema = EMA(model, ema_decay)
        ema.register()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        scheduler = timmscheduler.CosineLRScheduler(optimizer,
                                                    t_initial=num_epochs_repeat * num_steps_per_epoch,  # 第一次重启的迭代次数
                                                    lr_min=lr_min,  # 最小学习率
                                                    warmup_lr_init=warmup_lr_init,  # warmup初始学习率
                                                    warmup_t=warmup_epoch,
                                                    # noise_range_t = (5,30),
                                                    # noise_pct = 0.1,
                                                    # k_decay = 0.5,
                                                    # cycle_limit = 1,
                                                    t_in_epochs=False)

        # train_loss_list = train(epochs, model, train_loader, optimizer, scheduler, ema)

        train_loss_list = []
        optimizer.zero_grad()

        for epoch in tqdm(range(epochs)):
            num_updates = epoch * num_steps_per_epoch
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                # ==== inputs & mix inputs ====
                inputs = inputs.to(device)
                labels = labels.to(device)

                if Mixup_active:
                    mixed_inputs, mixed_labels = mixup_fn(inputs, labels)
                    outputs = model(mixed_inputs)
                    loss = criterion(outputs, mixed_labels)

                elif Facemixup_active:
                    mixed_inputs, mixed_labels = mixup_fn(inputs, labels)
                    # out = torchvision.utils.make_grid(mixed_inputs)
                    # tensor2img(out)
                    outputs_origin = model(inputs)
                    outputs_mixed = model(mixed_inputs)
                    origin_loss = criterion_none_reduction(outputs_origin, labels)
                    mixed_loss = criterion_none_reduction(outputs_mixed, mixed_labels)

                    # get iou
                    iou_list = tensor2box2iou(inputs, yolo)
                    iou_list = torch.tensor(iou_list)
                    # print(iou_list)

                    # facemix
                    overallloss = mixed_loss.sum() + ((1 - iou_list.cuda()) * (origin_loss + origin_loss.flip(0))).sum()
                    loss = overallloss / len(inputs)

                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step_update(num_updates=num_updates)
                ema.update()

                optimizer.zero_grad()
                running_loss += loss.item() * inputs.size(0)

            scheduler.step(epoch + 1)
            running_loss = running_loss / len(train_loader.dataset)
            print(f"======================= epoch:{epoch + 1}/{epochs} =======================")
            print(f"train loss:{running_loss}  lr:{optimizer.state_dict()['param_groups'][0]['lr']}")

            train_loss_list.append(running_loss)

        num_epochs = 50
        mu = 1
        coefficient_u = 5
        burnin_epochs = 0
        ema_teacher = True
        threshold = 0.7

        learning_rate = 1.5e-4
        dataset_train_x, dataset_train_u = build_dataset_ssl(numclass, traindir, anno_percent)

        sampler_train_x = torch.utils.data.RandomSampler(dataset_train_x)
        sampler_train_u = torch.utils.data.RandomSampler(dataset_train_u)

        data_loader_train_x = torch.utils.data.DataLoader(
            dataset_train_x, sampler=sampler_train_x,
            batch_size=batch_size * 2,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        data_loader_train_u = torch.utils.data.DataLoader(
            dataset_train_u, sampler=sampler_train_u,
            batch_size=int(batch_size * mu * 2),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        model_ema = None
        model_ema = ModelEma(model, decay=ema_decay, device='', resume='')
        model_ema.base_decay = model_ema.decay

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05, betas=(0.9, 0.95))
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)

        # semi train
        for epoch in tqdm(range(num_epochs)):
            model.train()
            optimizer.zero_grad()

            epoch_x = epoch * math.ceil(len(data_loader_train_u) / len(data_loader_train_x))
            data_iter_x = iter(data_loader_train_x)

            for inputs_u, labels_u in tqdm(data_loader_train_u):
                try:
                    inputs_x, labels_x = next(data_iter_x)
                except Exception:
                    epoch_x += 1
                    print(f"reshuffle data_loader_train_x at epoch_x={epoch_x}")
                    data_iter_x = iter(data_loader_train_x)
                    inputs_x, labels_x = next(data_iter_x)

                # x: labeled
                # u_w: unlabeled weak augment, 输入teacher,生成pseudo label
                # u_s: unlabeled strong augment, 如果peseudo label置信度大于阈值, (u_s , y_pseudo)作为训练样本
                # 最终只有高置信度的 pseudo label 才会贡献未标记样本损失

                inputs_x = inputs_x.to(device)
                labels_x = labels_x.to(device)
                inputs_u_w, inputs_u_s = inputs_u
                inputs_u_w = inputs_u_w.to(device)
                inputs_u_s = inputs_u_s.to(device)
                labels_u = labels_u.to(device)

                # labeled data
                outputs = model(inputs_x)
                loss_x = criterion(outputs, labels_x)

                # unlabeled data
                if epoch >= burnin_epochs:
                    if ema_teacher:
                        logits_u_w = model_ema.ema(inputs_u_w)
                    else:
                        logits_u_w = model(inputs_u_w)

                    # weak augment 生成 pseudo label
                    pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
                    max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(threshold).float()

                    logits_u_s = model(inputs_u_s)
                    loss_per_sample = F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none')
                    loss_u = (loss_per_sample * mask).mean()
                else:
                    loss_u = 0.

                loss = loss_x + coefficient_u * loss_u
                loss.backward()
                optimizer.step()
                model_ema.update(model)
                optimizer.zero_grad()
                torch.cuda.synchronize()

                loss_u_value, pseudo_acc = 0., 0.
                if epoch >= burnin_epochs:
                    loss_u_value = loss_u.item()
                    pseudo_acc_batch = (pseudo_targets_u == labels_u).float()
                    if mask.sum() > 0:
                        pseudo_acc = (pseudo_acc_batch * mask).sum() / mask.sum()

            print(f"======================= epoch:{epoch + 1}/{epochs} =======================")
            print(
                f"overall loss:{loss}  unlabel loss:{loss_u_value}  pseudo_acc:{pseudo_acc}  lr:{optimizer.state_dict()['param_groups'][0]['lr']}")

            valid_acc, valid_loss = validate(model, valid_loader, ema)
            print(f"Fold {fold} - Validation Accuracy: {valid_acc}, Validation Loss: {valid_loss}")


start = time.time()
k_fold_cross_validation(k=5, epochs=num_epochs, dataset=train_dataset)
end = time.time()
print(f"运行时间： {(end - start) / 60} min")