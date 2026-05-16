import os, shutil
from torch.nn import Module
from torch import optim
from torch.optim import lr_scheduler

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

def remove_ipynb_checkpoints(root_dir):
    """
    Tìm và xóa tất cả các thư mục có tên '.ipynb_checkpoints' 
    bắt đầu từ thư mục gốc root_dir.
    """
    # os.walk sẽ duyệt qua toàn bộ cây thư mục (bao gồm cả subfolders)
    for root, dirs, files in os.walk(root_dir):
        if ".ipynb_checkpoints" in dirs:
            # Xây dựng đường dẫn đầy đủ đến thư mục cần xóa
            folder_path = os.path.join(root, ".ipynb_checkpoints")
            
            try:
                # Sử dụng shutil.rmtree để xóa toàn bộ thư mục và nội dung bên trong
                shutil.rmtree(folder_path)
                print(f"Đã xóa: {folder_path}")
            except Exception as e:
                print(f"Lỗi khi xóa {folder_path}: {e}")


def select_optimizer(opt_name: str, lr: float, model: Module) -> optim.Optimizer:
    if opt_name == "adam":
        # print("opt_name: adam")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt

def select_scheduler(sched_name: str, opt: optim.Optimizer, hparam=None) -> lr_scheduler._LRScheduler:
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80, 90], gamma=0.1)
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler