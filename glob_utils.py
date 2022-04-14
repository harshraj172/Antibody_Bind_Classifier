import pandas as pd
import torch 

# Loss function
def task_loss(pred, target, criterion, use_mean=True):
    eps = 1e-10
    l1_loss = criterion(pred+eps, target)
    l2_loss = torch.sum(torch.abs(pred - target))
    if use_mean:
        l2_loss /= pred.shape[0]

    rescale_loss = l1_loss # train_dataset.norm2units(l1_loss, FLAGS.task)
    return l1_loss, l2_loss, rescale_loss
    
def metric(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 
    
    y_pred = (y_pred > 0.5).type(torch.uint8)  
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    
    result_df = pd.DataFrame([[precision.item(), recall.item(), f1.item()]], columns=['Precision', 'Recall', 'F1 Score'])
    return result_df
