import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake, device):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    batch_size = logits_fake.size()[0]

    zeros = torch.zeros((batch_size, 1)).to(device)
    loss1 = bce_loss(logits_fake, zeros)

    ones = torch.ones((batch_size, 1)).to(device)
    loss2 = bce_loss(logits_real, ones)

    loss = (loss1 + loss2) / batch_size
    
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake, device):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    batch_size = logits_fake.size()[0]
    ones = torch.ones((batch_size, 1)).to(device)
    loss = bce_loss(logits_fake, ones) / batch_size
    
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake, device):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    batch_size = scores_fake.size()[0]

    loss1 = torch.sum(scores_fake ** 2) / 2

    ones = torch.ones((batch_size, 1)).to(device)
    loss2 = torch.sum((scores_real - ones) ** 2) / 2

    loss = (loss1 + loss2) / batch_size
    
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake, device):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    batch_size = scores_fake.size()[0]
    ones = torch.ones((batch_size, 1)).to(device)
    loss = torch.sum((scores_fake - ones) ** 2) / (2 * batch_size)
    
    ##########       END      ##########
    
    return loss







