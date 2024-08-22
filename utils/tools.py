# seed
import torch
import random
import numpy as np
import os
from ptflops import get_model_complexity_info
import GPUtil

from torch import inf  # 2.0
# from torch._six import inf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2


os.environ['KMP_DUPLICATE_LIB_OK']='True'



class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def seed_everything(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)
    print("========== seed set ==========")



# 参数量
def count_parameters(model):
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    memory = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / 1024 / 1024
    print(f"flops: {flops}  params: {params}  memory: {memory} MB")


# 内存计算
def count_memory():
    gpu = GPUtil.getGPUs()[0]
    print(f"Free memory: {gpu.memoryFree} MB")



# convert
def tensor2np(inputtensor, mean, std):
    inputtensor = inputtensor.permute(1, 2, 0).detach().cpu().numpy()  # 变成 [224, 224, 3]
    inputtensor = std * inputtensor + mean
    inputtensor = (inputtensor * 255).astype(np.uint8)
    return inputtensor

def np2img(inputnp):
    plt.imshow(inputnp, cmap='gray')
    plt.show()


def np2tensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256





def tensor2box(inputtensor,x1,y1,x2,y2):
    imgarray = inputtensor.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(imgarray)

    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1, edgecolor='r',facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.show()


def getiou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection

    iou = intersection / union
    return iou