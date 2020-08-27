import torch
import numpy as np
from functools import partial
import scipy.sparse as ss
import sklearn.metrics as skmetrics
from torch.functional import F
from sklearn.utils import sparsefuncs
from poutyne.utils import torch_to_numpy

feps = 1e-8  # np.finfo(float).eps


def _convert_to_numpy(*args):
    r"""Convert a list of array to numpy if they are torch tensors"""
    for arr in args:
        if isinstance(arr, torch.Tensor):
            yield torch_to_numpy(arr)
        else:
            yield arr


def _assert_type(*args, dtype=torch.Tensor, ign_none=True):
    r"""Verify if the set of argument are from the same type"""
    return all([((ign_none and x is None) or isinstance(x, dtype)) for x in args])


def __weight_normalizer(weights, y_pred):
    r"""Normalize the weights, or create the weights if a None is fed"""
    if weights is None:
        weights = torch.ones_like(y_pred)
    else:
        weights = torch.Tensor(weights)

    weights /= torch.sum(weights, dim=0)
    return weights


def pearsonr(y_pred, y_true, per_tasks=False):
    r"""
    Compute the Pearson's r metric between y_pred and y_true.
    y_pred and y_true have to be from the same type and the same shape. 

    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. 

    Arguments
    ----------
        y_pred: iterator
            Predicted values
        y_true: iterator
            Ground truth
        per_tasks: bool, optional
            Whether to compute metric per task in the output
            (Default, value = None)

    Returns
    -------
        The Pearson correlation coefficient between y_pred and y_true

    See Also
    --------
        `gnnpooling.utils.metrics.pearsonr_square`, `gnnpooling.utils.metrics.r2_score`,  
    """
    if _assert_type(y_pred, y_true, dtype=torch.Tensor):
        mean_y_pred = torch.mean(y_pred, dim=0)
        mean_y_true = torch.mean(y_true, dim=0)
        y_predm = y_pred - mean_y_pred
        y_truem = y_true - mean_y_true
        r_num = torch.sum(y_predm * y_truem, dim=0)
        r_den = torch.norm(y_predm, 2, dim=0) * torch.norm(y_truem, 2, dim=0)
        r = r_num / (r_den + feps)
        r = torch.clamp(r, -1.0, 1.0)
        if not per_tasks:
            r = r.mean()
        return r
    return __np_pearsonr(y_pred, y_true, per_tasks)


def r2_score(y_pred, y_true, per_tasks=False):
    r"""
    Compute the R-square metric  between y_pred ( :math:`\hat{y}`) and y_true (:math:`y`):

    .. math:: R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}

    y_pred and y_true have to be from the same type and the same shape.

    Arguments
    -----------
        y_pred: iterator
            predicted values
        y_true: iterator
            ground truth
        per_tasks: bool, optional
            Whether to compute R2 for each task
            (Default, value = None)

    Returns
    -------
        The R2 between y_pred and y_true
    """
    if _assert_type(y_pred, y_true, dtype=torch.Tensor):
        mean_y_true = torch.mean(y_true, dim=0)
        ss_tot = torch.sum(torch.pow(y_true.sub(mean_y_true), 2), dim=0)
        ss_res = torch.sum(torch.pow(y_pred - y_true, 2), dim=0)
        r2 = 1 - (ss_res / (ss_tot + feps))
        if not per_tasks:
            r2 = r2.mean()
        return r2
    return __np_r2(y_pred, y_true, per_tasks)


def mse(y_pred, y_true, weights=None, per_tasks=False):
    r"""
    Compute the mean square error between y_pred and y_true

    .. math::
        MSE = \frac{\sum_i^N (y_i - \hat{y}_i)^2}{N} 

    y_pred, y_true (and weights) need to have the same type and the same shape.
    y_pred and y_true have to be from the same type and the same shape.

    Arguments
    ----------
        y_pred: iterator
            Predicted values
        y_true: iterator
            Ground truth
        weigths: iterator, optional
            Weights for each example (for each task). The weights are normalized so that their sum is 1.
            Using boolean values, the weights can act as a mask. 
            All examples have a weight of 1 by default
        per_tasks: bool, optional
            Whether to compute error per task
            (Default, value = None)

    Returns
    -------
        mse: mse between y_pred and y_true

    """
    if _assert_type(y_pred, y_true, dtype=torch.Tensor):
        weights = __weight_normalizer(weights, y_pred)
        #res = torch.mean(torch.pow(y_pred - y_true, 2), dim=0)
        res = torch.sum(weights * torch.pow(y_pred - y_true, 2), dim=0)
        if not per_tasks:
            res = res.mean()
        return res
    return __np_mse(y_pred, y_true, per_tasks=per_tasks, weights=weights)


def accuracy(y_pred, y_true, weights=None, per_tasks=False, threshold=0.5):
    r"""
    Compute the prediction accuracy given the true class labels y_true
    and predicted values y_pred. Note that y_true and y_pred should
    both be array of binary values and have the same shape

    Arguments
    ----------
        y_pred: iterator
            Predicted labels
        y_true: iterator
            True class labels
        weigths: iterator, optional
            Weights for each example (for each task). 
        Using boolean values, the weights can act as a mask. 
        All examples have a weight of 1 by default.
            (Default, value = None)
        per_tasks: bool, optional
            Compute accuracy per task
            (Default, value = None)
        threshold : float, optional
            Minimum threshold to transform prediction into binary
            (Default value = 0.5)

    Returns
    -------
        the classification accuracy
    """
    y_pred, y_true = _convert_to_numpy(y_pred, y_true)
    if y_pred.shape != y_true.shape and len(y_true) == len(y_pred):
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = (np.asarray(y_pred) > threshold).astype(int)
    y_true = np.asarray(y_true)

    if per_tasks and len(y_pred.shape) == len(y_pred.shape) == 2:
        n_classes = y_pred.shape[-1]
        if weights is None:
            weights = [None] * n_classes
        return [skmetrics.accuracy_score(y_true[:, n], y_pred[:, n], sample_weight=weights[n]) for n in range(n_classes)]

    if weights is not None:
        weights = weights.flatten()
    return skmetrics.accuracy_score(y_true.flatten(), y_pred.flatten(), sample_weight=weights)


def f1_score(y_pred, y_true, per_tasks=False, threshold=0.5, average='weighted'):
    r"""
    Compute the f1_score given the true class labels and predicted labels.
    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. If the computation is done on different tasks, then the dimension 0 represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks.

    Arguments
    ----------
        y_pred: iterator
            Predicted values
        y_true: iterator
            True class labels
        per_tasks: bool, optional
            Compute the f1_score for each task separately
            (Default, value = None)
        threshold: float, optional
            Minimum threshold to transform prediction into binary
            (Default value = 0.5)
        average: string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted'], optional
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
    Returns
    -------
        F1 scores, given each class for the multiclass task.

    """
    y_pred, y_true = _convert_to_numpy(y_pred, y_true)
    y_pred = (np.asarray(y_pred) > threshold).astype(int)
    if per_tasks:
        return skmetrics.f1_score(y_true, y_pred, average=None)
    try:
        score = skmetrics.f1_score(y_true, y_pred, average=average)
    except:
        if average != None:
            not_all_zeros = np.sum(y_true, axis=1) > 0
            y_true = y_true[not_all_zeros, :]
            y_pred = y_pred[not_all_zeros, :]
        score = skmetrics.f1_score(y_true, y_pred, average=average)
    return score


def roc_auc_score(y_pred, y_true, per_tasks=False, average=None):
    r"""
    Compute the roc_auc_score given the true class labels and predicted labels.
    By setting per_tasks to True, the metric will be computed
    for each task independently.

    Arguments
    ----------
        y_pred: iterator
            Predicted values
        y_true: iterator
            True class labels
        per_tasks: bool, optional
            Compute the roc_auc_score for each task separately
            (Default, value = None)
        average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                           'weighted']
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
    Returns
    -------
        ROC-AUC score, given each class for the multiclass task.

    """
    y_pred, y_true = _convert_to_numpy(y_pred, y_true)
    if y_pred.shape != y_true.shape and len(y_true) == len(y_pred):
        _y_true = np.zeros_like(y_pred)
        _y_true[np.arange(len(y_true)), y_true] = 1
        y_true = _y_true
    y_true = np.asarray(y_true)

    if per_tasks:
        return skmetrics.roc_auc_score(y_true, y_pred, average=None)

    try:
        score = skmetrics.roc_auc_score(y_true, y_pred, average=average)
    except:
        if average != None:
            not_all_zeros = np.sum(y_true, axis=1) > 0
            y_true = y_true[not_all_zeros, :]
            y_pred = y_pred[not_all_zeros, :]
        score = skmetrics.roc_auc_score(y_true, y_pred, average=average)
    return score


def multilabel_confusion_matrix(y_pred, y_true, sample_weight=None,
                                samplewise=False):
    r"""
    Compute a confusion matrix for each class or sample in the output. 
    If `sample_weight` is provided, it is required to have the same shape as y_pred
    This function is inspired by issue #11179 on the sklearn github
    :see: https://github.com/scikit-learn/scikit-learn/pull/11179
    and related pull request for a detailled overview

    Arguments
    ----------
        y_pred: numpy.ndarray
            predicted value
        y_true: numpy.ndarray
            ground truth
        sample_weight: numpy.ndarray, optional
            sample weights for each 
            example in the output. 
        samplewise: bool optional
            Whether to compute the confusion matrix 
            samplewise or taskwise

    Returns
    -------
        a confusion matrix of size (N, 2, 2), where N is either the number of labels 
        or the total number of samples.

    """
    y_true = ss.csr_matrix(y_true)
    y_pred = ss.csr_matrix(y_pred)
    sum_axis = 1 if samplewise else 0

    # calculate weighted counts
    true_and_pred = y_true.multiply(y_pred)
    tp_sum = sparsefuncs.count_nonzero(true_and_pred, axis=sum_axis,
                                       sample_weight=sample_weight)
    pred_sum = sparsefuncs.count_nonzero(y_pred, axis=sum_axis,
                                         sample_weight=sample_weight)
    true_sum = sparsefuncs.count_nonzero(y_true, axis=sum_axis,
                                         sample_weight=sample_weight)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum

    if sample_weight is not None and samplewise:
        sample_weight = np.array(sample_weight)
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
        tn = sample_weight * y_true.shape[1] - tp - fp - fn
    elif sample_weight is not None:
        tn = sum(sample_weight) - tp - fp - fn
    elif samplewise:
        tn = y_true.shape[1] - tp - fp - fn
    else:
        tn = y_true.shape[0] - tp - fp - fn

    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


def loss_fn(pred, targets, weights=None, base_criterion=None):
    if weights is not None:
        base_criterion.reduction = 'none'
        loss = base_criterion(pred, targets) * weights.detach()
        return loss.mean()
    return base_criterion(pred, targets)


def get_loss(dataset):
    if dataset.upper() in ['TOX21', 'FRAGMENTS', 'ALERTS']:
        return partial(loss_fn, base_criterion=torch.nn.BCEWithLogitsLoss(reduction='mean'))
    else:
        return partial(loss_fn, base_criterion=torch.nn.CrossEntropyLoss(reduction='mean'))
