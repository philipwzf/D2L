import torch
from d2l import torch as d2l

class Classifier(d2l.Module):
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss',self.loss(Y_hat,batch[-1]),train=False)
        self.plot('acc',self.accuracy(Y_hat,batch[-1],train=False))
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),lr=self.lr)
    
    def accuracy(self,Y_hat,Y,averaged=True):
        """Compute the number of correct predictions"""

        #Y_hat is the big matirx that stores the prediction scores for all the samples
        #we split y_hat into two columns - I dont know why we need this tho
        Y_hat = Y_hat.reshape((-1,Y_hat.shape[-1]))

        #use argmax to obtain the best category for each obj
        preds = Y_hat.argmax(axis=1).type(Y.dtype)

        #compare if the category is the same as what the truth is. If yes, then 1; no then 0
        compare = (preds==Y.reshape(-1)).type(torch.float32)

        #calculate the num of correct cases/all cases
        return compare.mean() if averaged else compare