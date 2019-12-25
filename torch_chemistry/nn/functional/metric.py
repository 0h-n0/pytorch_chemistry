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
    first = torch.tensor([0],
                         dtype=_dtype, device=_device)
    threshold_idxs = torch.cat([torch.where(diff(sorted_y_score))[0],
                                last])
    threshold_idxs = torch.cat([first,
                                threshold_idxs])
    tps = torch.cumsum(sorted_y_true, dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = torch.cat([first, tps])
    print(len(fps), len(tps))
    fps = torch.cat([first, fps])
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return fpr, tpr, threshold_idxs


def auc(x, y):
    return torch.trapz(y, x)


def roc_auc_score(y_score, y_true):
    fpr, tpr, _ = roc_curve(y_score, y_true)
    return auc(fpr, tpr)
