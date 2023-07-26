import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

def softmax(X):
    #get the exp for each term
    X_exp = torch.exp(X)

    #calculate the sum of each row
    partition = X_exp.sum(1,keepdim=True)

    #return the exp/sum of exp to make sure each row sums up to 1
    return X_exp/partition

class SoftmaxRegressionScratch(d2l.Classifier):
    """The Softmax Classification model implemented from scratch"""
    def __init__(self,num_inputs,num_outputs,lr,sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0,sigma,size=(num_inputs,num_outputs),requires_grad=True)
        self.b = torch.zeros(num_outputs,requires_grad=True)

    def parameters(self):
        return [self.W,self.b]
    
    def forward(self,X):
        #flatten the 28x28 pixel image in the batch into a 1d vector
        X = X.reshape((-1,self.W.shape[0]))
        return softmax(torch.matmul(X,self.W)+self.b)
    
    def loss(self,y_hat,y):
        return cross_entropy(y_hat,y)
    
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[list(range(len(y_hat))),y]).mean()

if __name__ == '__main__':
    data = d2l.FashionMNIST(batch_size=256)
    model = SoftmaxRegressionScratch(num_inputs=784,num_outputs=10,lr=0.1)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model,data)
    
    X,y = next(iter(data.val_dataloader()))
    preds = model(X).argmax(axis = 1)
    wrong = preds.type(y.dtype) != y
    X,y,preds = X[wrong],y[wrong],preds[wrong]
    labels = [a+'\n'+b for a,b in zip(data.text_labels(y),data.text_labels(preds))]
    data.visualize([X,y],labels = labels)
    plt.show()
