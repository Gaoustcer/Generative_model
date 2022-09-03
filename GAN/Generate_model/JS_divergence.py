from torch.nn import KLDivLoss
import torch
def JSdivergence(inputa:torch.tensor,inputb:torch.tensor):
    KLdiv = KLDivLoss()
    assert inputa.shape == inputb.shape

    return torch.div((KLdiv(inputa,inputb) + KLdiv(inputb,inputa)),2)