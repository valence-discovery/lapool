import torch
import numpy as np
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


def __np_pearsonr(y_pred, y_true, per_tasks=True):
    r"""Compute Pearson'r from numpy.ndarray inputs"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean_y_pred = np.mean(y_pred, axis=0)
    mean_y_true = np.mean(y_true, axis=0)
    y_predm = y_pred - mean_y_pred
    y_truem = y_true - mean_y_true
    r_num = np.sum(y_predm * y_truem, axis=0)
    r_den = np.linalg.norm(y_predm, 2, axis=0) * \
        np.linalg.norm(y_truem, 2, axis=0)
    r = r_num / (r_den + feps)
    r = np.clip(r, -1.0, 1.0)
    if not per_tasks:
        return np.mean(r)
    return r


def __np_r2(y_pred, y_true, per_tasks=True, weights=None):
    r"""Compute R2 from numpy.ndarray inputs"""
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    if per_tasks:
        return skmetrics.r2_score(y_true, y_pred, sample_weight=weights, multioutput='raw_values')
    return skmetrics.r2_score(y_true, y_pred, sample_weight=weights)


def __std_se(y_pred, y_true, per_tasks=True, weights=None):
    r"""Compute the std from numpy.ndarray inputs"""
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    # se : squared_error
    se = (y_pred - y_true) ** 2
    if weights is None:
        std = np.std(se, axis=0, ddof=1)
    else:
        weights = np.array(__weight_normalizer(weights, y_pred))
        n = y_pred.shape[0]
        mean_se = np.sum(weights * se, axis=0)
        dividor = (n - 1) / n
        var_se = np.sum(weights * np.power(se - mean_se, 2), axis=0) / dividor
        std = np.sqrt(var_se)
    if per_tasks:
        return std
    return np.mean(std)


def __np_mse(y_pred, y_true, per_tasks=True, weights=None):
    r"""Compute the MSE from numpy.ndarray inputs"""
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    errors = np.average((y_pred - y_true) ** 2, axis=0, weights=weights)
    if not per_tasks:
        return np.mean(errors)
    return errors


def __np_mae(y_pred, y_true, per_tasks=True, weights=None):
    r"""Compute the MAE from numpy.ndarray inputs"""
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    errors = np.average(np.abs(y_pred - y_true), axis=0, weights=weights)
    if not per_tasks:
        return np.mean(errors)
    return errors


def __weight_normalizer(weights, y_pred):
    r"""Normalize the weights, or create the weights if a None is fed"""
    if weights is None:
        weights = torch.ones_like(y_pred)
    else:
        weights = torch.Tensor(weights)

    weights /= torch.sum(weights, dim=0)
    return weights


def masked_soft_margin_loss(y_pred, y_true, mask=None, per_tasks=False):
    r"""
    Compute the masked soft margin loss.

    .. caution::
        This is kept for compatibility and could be removed in later version. 
        See pytorch `multilabel_soft_margin_loss` implementation with weights
        and different reduction strategy.

    Arguments
    ----------
        y_pred: torch.Tensor
            logits from network output
        y_true: torch.Tensor
            True label for the classification task.
        mask: torch.Tensor, optional
            weight/mask covering the output.
        per_tasks: bool, optional
            whether to return loss for each task independently
            (Default, value = False)

    Returns
    -------
        loss metric according the set of arguments

    """
    loss = - (y_true * F.logsigmoid(y_pred) +
              (1 - y_true) * F.logsigmoid(-y_pred))
    if mask is not None:
        loss *= mask
    if not per_tasks:
        loss = loss.sum(dim=1)
    return torch.mean(loss, dim=0)


def pearsonr(y_pred, y_true, per_tasks=False):
    r"""
    Compute the Pearson's r metric between y_pred and y_true.
    y_pred and y_true have to be from the same type and the same shape. 

    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. The dimension 0 represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks.

    .. caution:: 
        THIS IS NOT THE SQUARE-ROOT OF R2_SCORE! The pearsonr measures the 
        linearity between y_pred and y_true, but the r2_score measures the 
        variance between two set of data. They are only equivalent if y_true 
        is the optimal linear model that fits y_pred. 

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


def pearsonr_squared(y_pred, y_true, per_tasks=False):
    r"""
    Compute the squared Pearson's r metric between y_pred and y_true.
    y_pred and y_true have to be from the same type and the same
    shape. 

    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. The dimension 0 represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks.

    .. caution:: 
        Although it is readily believed that the R2 is the square of the pearson'r, 
        this is only true depending on how it is computed. Here we are reporting the 
        `variation explained squared` (intercept assumed) and not the `variance explained`. 
        In layman terms, the pearsonr measures the linearity between y_pred and y_true, 
        but the r2_score measures the variance between two set of data. 
        They are only equivalent if y_pred is the optimal linear model that fits y_true, 
        as should normally be the case in linear regression. For more information, please see: 
        https://stats.stackexchange.com/q/7357/ and https://stats.stackexchange.com/q/26176/

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
        The square of the Pearson correlation coefficient

    See Also
    --------
        `gnnpooling.utils.metrics.pearsonr`, `gnnpooling.utils.metrics.r2_score`,  
    """

    pr = pearsonr(y_pred, y_true, per_tasks=per_tasks)
    return pr ** 2


def r2_score(y_pred, y_true, per_tasks=False):
    r"""
    Compute the R-square metric  between y_pred ( :math:`\hat{y}`) and y_true (:math:`y`):

    .. math:: R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}

    y_pred and y_true have to be from the same type and the same shape.
    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. The dimension 0 represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks.

    .. caution:: 
        Although it is readily believed that the R2 is the square of the pearson'r, 
        this is only true depending on how it is computed. Here we are reporting the 
        `variance explained` (intercept assumed) and not the `variation explained`. 
        In layman terms, the pearsonr measures the linearity between y_pred and y_true, 
        but the r2_score measures the variance between two set of data. 
        They are only equivalent if y_pred is the optimal linear model that fits y_true, 
        as should normally be the case in linear regression. For more information, please see: 
        https://stats.stackexchange.com/q/7357/ and https://stats.stackexchange.com/q/26176/

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

    See Also
    --------
        `gnnpooling.utils.metrics.pearsonr`
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


def std_se(y_pred, y_true, weights=None, per_tasks=False):
    r"""
    Compute the standard deviation of the square error between y_pred and y_true 
    y_pred and y_true have to be from the same type and the same shape.

    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. The dimension 0 represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks.

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
            Whether to compute metric per task in the output
            (Default, value = None)

    Returns
    -------
        The std of the mean square error between y_pred and y_true

    See Also
    --------
        `gnnpooling.utils.metrics.mse`, `gnnpooling.utils.metrics.mae`
    """
    if _assert_type(y_pred, y_true, weights, dtype=torch.Tensor):
        weights = __weight_normalizer(weights, y_pred)
        n = y_pred.shape[0]
        # se : squared_error
        se = (y_pred - y_true) ** 2
        mean_se = torch.sum(weights * se, dim=0)
        dividor = (n - 1) / n
        var_se = torch.sum(
            weights * torch.pow(se - mean_se, 2), dim=0) / dividor
        std = torch.sqrt(var_se)
        if not per_tasks:
            std = std.mean()
        return std
    return __std_se(y_pred, y_true, per_tasks=per_tasks, weights=weights)


def vse(y_pred, y_true, weights=None, per_tasks=False):
    r"""
    Compute the variance of the square error between y_pred and y_true y_pred 
    and y_true have to be from the same type and the same shape.

    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. The dimension 0 represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks.

    Arguments
    ----------
        y_pred: iterator
            predicted values
        y_true: iterator
            ground truth
        weigths: iterator, optional
            Weights for each example (for each task). The weights are normalized so that their sum is 1.
            Using boolean values, the weights can act as a mask. 
            All examples have a weight of 1 by default
        per_tasks: bool, optional
            Compute metric per task in the output
            (Default, value = False)

    Returns
    -------
        vse: vse between y_pred and y_true

    See Also
    --------
        `gnnpooling.utils.metrics.std_se`, `gnnpooling.utils.metrics.mse`, `gnnpooling.utils.metrics.mae`
    """
    std = std_se(y_pred, y_true, weights=weights, per_tasks=per_tasks)
    return std ** 2


def mse(y_pred, y_true, weights=None, per_tasks=False):
    r"""
    Compute the mean square error between y_pred and y_true

    .. math::
        MSE = \frac{\sum_i^N (y_i - \hat{y}_i)^2}{N} 

    y_pred, y_true (and weights) need to have the same type and the same shape.
    y_pred and y_true have to be from the same type and the same shape.

    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. The dimension 0 represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks.

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

    See Also
    --------
        `gnnpooling.utils.metrics.mae`, `gnnpooling.utils.metrics.std_se`,  
        `gnnpooling.utils.metrics.vse`, `gnnpooling.utils.metrics.roc_auc_score`,
        `gnnpooling.utils.metrics.accuracy`
    """
    if _assert_type(y_pred, y_true, dtype=torch.Tensor):
        weights = __weight_normalizer(weights, y_pred)
        #res = torch.mean(torch.pow(y_pred - y_true, 2), dim=0)
        res = torch.sum(weights * torch.pow(y_pred - y_true, 2), dim=0)
        if not per_tasks:
            res = res.mean()
        return res
    return __np_mse(y_pred, y_true, per_tasks=per_tasks, weights=weights)


def mae(y_pred, y_true, weights=None, per_tasks=False):
    r"""
    Compute the mean absolute error between y_pred and y_true:

    .. math::
        MAE = \frac{\sum_i^N \abs{y_i - \hat{y}_i}}{N}

    y_pred, y_true (and weights) need to have the same type and the same shape.
    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. The dimension 0 represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks. y_pred and y_true have to be from the same type and the same shape.

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
            Whether to compute errors per task
            (Default, value = None)

    Returns
    -------
        mae: mae between y_pred and y_true

    See Also
    --------
        `gnnpooling.utils.metrics.mse`, `gnnpooling.utils.metrics.std_se`,  
        `gnnpooling.utils.metrics.vse`, `gnnpooling.utils.metrics.roc_auc_score`,
        `gnnpooling.utils.metrics.accuracy`
    """
    if _assert_type(y_pred, y_true, weights, dtype=torch.Tensor):
        weights = __weight_normalizer(weights, y_pred)
        res = torch.sum(weights * torch.abs(y_pred - y_true), dim=0)
        if not per_tasks:
            res = res.mean()
        return res
    return __np_mae(y_pred, y_true, per_tasks, weights=weights)


def accuracy(y_pred, y_true, weights=None, per_tasks=False, threshold=0.5):
    r"""
    Compute the prediction accuracy given the true class labels y_true
    and predicted values y_pred. Note that y_true and y_pred should
    both be array of binary values and have the same shape

    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. The dimension 0 represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks.

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

    See Also
    --------
        `gnnpooling.utils.metrics.f1_score`, `gnnpooling.utils.metrics.roc_auc_score`
    """
    y_pred, y_true = _convert_to_numpy(y_pred, y_true)
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

    See Also
    --------
        `gnnpooling.utils.metrics.mae`, `gnnpooling.utils.metrics.std_se`, 
        `gnnpooling.utils.metrics.vse`, `gnnpooling.utils.metrics.roc_auc_score`,
        `gnnpooling.utils.metrics.accuracy`
    """
    y_pred, y_true = _convert_to_numpy(y_pred, y_true)
    y_pred = (np.asarray(y_pred) > threshold).astype(int)
    
    if per_tasks:
        return skmetrics.f1_score(y_true, y_pred, average=None)
    return skmetrics.f1_score(y_true, y_pred, average=average)


def roc_auc_score(y_pred, y_true, per_tasks=False, average="macro"):
    r"""
    Compute the roc_auc_score given the true class labels and predicted labels.
    By setting per_tasks to True, the metric will be computed
    for each task independently. Otherwise the mean on all tasks will
    be returned. If the computation is done on different tasks, then the dimension 0 
    represents the data on which the metric is computed and
    dimensions 1, 2, 3... represent the different tasks.

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

    See Also
    --------
        `gnnpooling.utils.metrics.mae`, `gnnpooling.utils.metrics.std_se`, 
        `gnnpooling.utils.metrics.vse`, `gnnpooling.utils.metrics.f1_score`
        `gnnpooling.utils.metrics.accuracy`
    """
    y_pred, y_true = _convert_to_numpy(y_pred, y_true)
    
    if per_tasks:
        return skmetrics.roc_auc_score(y_true, y_pred, average=None)
    return skmetrics.roc_auc_score(y_true, y_pred, average=average)


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
