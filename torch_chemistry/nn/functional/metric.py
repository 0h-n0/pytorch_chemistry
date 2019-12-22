import torch


def diff(x):
    return x[1:] - x[:-1]

def roc_curve(y_score, y_true):
    sorted_y_score, indices = torch.sort(y_score)
    uniques = torch.unique(y_true)
    sorted_y_true = y_true[indices]
    _device = y_true.device
    _dtype = y_true.dtype
    last = torch.tensor([sorted_y_true.shape[0] - 1],
                        dtype=_dtype, device=_device)
    threshold_idxs = torch.cat([torch.where(diff(sorted_y_score))[0],
                                last])
    tps = torch.cumsum(sorted_y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps


    return fps, tps, soreted_y_score[threshold_idxs]


def auc():
    return


def roc_auc_score(y_score, y_true):
    fps, tps, _ = roc_curve(y_score, y_true)
    torch.trapz(
    return
