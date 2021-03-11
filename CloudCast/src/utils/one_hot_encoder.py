import torch

#adapted from http://jacobkimmel.github.io/pytorch_onehot/
def make_one_hot(labels, C=4):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''

    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(1),  labels.size(2), labels.size(3)).zero_()
    return one_hot.scatter_(1, labels.data.unsqueeze(1), 1)



def make_one_hot_reduced(labels, C=4):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''

    one_hot = torch.FloatTensor(C, labels.size(0),  labels.size(1)).zero_()
    return one_hot.scatter_(0, labels.long().data.unsqueeze(0), 1)




