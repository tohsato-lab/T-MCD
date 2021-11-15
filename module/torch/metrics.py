import numpy as np
import torch
from pytorch_lightning.metrics.functional import iou
from sklearn.metrics import recall_score

SMOOTH = 1e-15


def categorical_accuracy(pred_y, true_y):
    pred_y = pred_y.max(dim=1)[1]
    acc = pred_y.eq(true_y.data).float().mean().item()
    return acc


def binary_accuracy(pred_y, true_y):
    assert true_y.size() == pred_y.size(), "pred_y:{}, true_y,{}".format(pred_y.shape, true_y.shape)
    return (pred_y > 0.5).eq(true_y.data).float().mean().item()


def mean_accuracy(pred_y, true_y, mode='mean'):
    """
    :param pred_y:
    :param true_y:
    :param mode:
    :return:
    """
    score_list = []
    pred_y = pred_y.max(dim=1)[1].cpu().numpy()
    true_y = true_y.cpu().numpy()
    for pred, true in zip(pred_y, true_y):
        m_acc_scores = []
        for label_id in np.unique(true):
            p_bool_tensor = np.where(pred == label_id, True, False)
            t_bool_tensor = np.where(true == label_id, True, False)
            common_tensor = (p_bool_tensor * t_bool_tensor)
            # mean_acc
            m_acc = np.sum(common_tensor) / np.sum(t_bool_tensor)
            m_acc_scores.append(m_acc)
        score_list.extend(m_acc_scores)
    if len(score_list) == 0:
        score_list.append(0)
    return np.nanmean(score_list) if mode == "mean" else score_list


def __fast_hist__(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def calc_hist(pred_y: torch.Tensor, true_y: torch.Tensor, n_class: int):
    """
    混合配列の生成
    :param pred_y:
    :param true_y:
    :param n_class:
    :param _:
    :return:
    """
    pred_y = pred_y.max(dim=1)[1].view(-1).cpu().numpy()
    true_y = true_y.view(-1).cpu().numpy()
    """not_background_idxes = np.where(true_y != background_id)
    predict = pred_y[not_background_idxes]
    label = true_y[not_background_idxes]"""
    if len(true_y) == 0:
        return np.zeros((n_class, n_class))
    return __fast_hist__(true_y.flatten(), pred_y.flatten(), n_class)


def __per_class_iou__(hist):
    return np.diag(hist) * 100 / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))


def iou_lightning_metrics(pred_y, true_y, mode='elementwise_mean', ignore_id=0):
    """
    :param ignore_id:
    :param pred_y:
    :param true_y:
    :param mode:
    :return:
    """
    pred_y = pred_y.max(dim=1)[1].view(-1).cpu()
    true_y = true_y.view(-1).cpu()
    return iou(pred_y, true_y, reduction=mode, ignore_index=ignore_id).numpy()


def iou_metrics(hist: np.ndarray, mode='mean', background_id=0):
    """
    混合行列からmIoUを計算
    !注意! 計算誤差で結果がずれるので、histは全てのデータのものを利用すべし。
    :param background_id:
    :param hist:
    :param mode: mean, none
    :return:
    """
    gt_per_class = hist.sum(axis=1)
    used_class_id_list = np.where(gt_per_class != 0)[0]
    hist = hist[used_class_id_list][:, used_class_id_list]
    iou_list = __per_class_iou__(hist)
    iou_list[background_id] = np.nan
    if mode == 'mean':
        return np.nanmean(iou_list)
    else:
        result = np.full((len(gt_per_class)), np.nan)
        result[used_class_id_list] = iou_list
        return result


def recall_metrics(pred_y, true_y, mode='mean', ignore_id: list = None):
    """
    :param ignore_id:
    :param pred_y:
    :param true_y:
    :param mode:
    :return:
    """
    score_list = []
    num_class = pred_y.shape[1]
    pred_y = pred_y.max(dim=1)[1].view(-1).cpu()
    true_y = true_y.view(-1).cpu()
    for label_id in range(num_class):
        if ignore_id is not None and label_id in ignore_id:
            continue
        bin_pred_y = pred_y.eq(label_id).int().numpy()
        bin_true_y = true_y.eq(label_id).int().numpy()
        score = recall_score(bin_true_y, bin_pred_y)
        score_list.append(score if score > SMOOTH else 0)
    return np.nanmean(score_list) if mode == "mean" else score_list


def dice_metrics(pred_y, true_y, mode='mean', ignore_id: list = None):
    """
    :param ignore_id:
    :param mode:
    :param pred_y:
    :param true_y:
    :return:
    """
    score_list = []
    num_class = pred_y.shape[1]
    pred_y = pred_y.max(dim=1)[1].view(-1).cpu()
    true_y = true_y.view(-1).cpu()
    for label_id in range(num_class):
        if ignore_id is not None and label_id in ignore_id:
            continue
        bin_pred_y = pred_y.eq(label_id).int()
        bin_true_y = true_y.eq(label_id).int()
        if bin_true_y.sum() == 0 and mode == "mean":
            continue
        bin_common = bin_pred_y * bin_true_y
        score = ((2 * bin_common.sum()) / (bin_pred_y.sum() + bin_true_y.sum() + SMOOTH)).item()
        score_list.append(score if score > SMOOTH else 0)
    return np.nanmean(score_list) if mode == "mean" else score_list
