import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn 
from d2l import torch as d2l
import collections
import inspect
from IPython import display

class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                        mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        if self.fig is None:
            self.fig = d2l.plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(d2l.plt.plot([p.x for p in v], [p.y for p in v],
                                        linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else d2l.plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)

class Module(nn.Module, d2l.HyperParameters):
    def __init__(self,plot_train_per_epoch=2,plot_valid_per_epoch=1):
        #Call the constructor of the superclass (nn?)
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self,y_hat,y):
        raise NotImplementedError
    
    def forward(self,X):
        assert hasattr(self,'net'), 'Newral network is defined'
        return self.net(X)
    
    def plot(self,key,value,train):
        """Plot a point in animation"""
        assert hasattr(self,'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx/self.trainer.num_train_batches
            n = self.trainer.num_train_batches/self.plot_train_per_epoch
        else:
            x = self.trainer.epoch+1
            n = self.trainer.num_val_batches/self.plot_valid_per_epoch
        self.board.draw(x,value.to(d2l.cpu()).detach().numpy(),('train_' if train else 'val_')+key,every_n = int(n))

    def training_step(self,batch):
        #the first argument is the value up till (excluding) the last element,
        #the second argument batch[-1] is the last element
        l = self.loss(self(*batch[:-1]),batch[-1])
        self.plot('loss',l,train=True)
        return l
    
    def validation_step(self,batch):
        l = self.loss(self(*batch[:-1]),batch[-1])
        self.plot('loss',l,train=False)

    def configure_optimizers(self):
        raise NotImplementedError

class DataModule(d2l.HyperParameters):
    """The base class of data"""
    def __init__(self,root='../data',num_workers=4):
        self.save_hyperparameters()

    def get_tensorloader(self,tensors,train,indices=slice(0,None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset,self.batch_size,shuffle=train)
    
    def get_dataloader(self,train):
        raise NotImplementedError
    
    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    def val_dataloader(self):
        return self.get_dataloader(train=False)
        
class Trainer(d2l.HyperParameters):
    def __init__(self,max_epochs,num_gpus=0,gradient_clip_val = 0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self,data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader if self.val_dataloader is not None else 0))

    def prepare_model(self,model):
        model.trainer = self
        model.board.xlim = [0,self.max_epochs]
        self.model = model

    def fit(self,model,data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch

    def fit_epoch(self):
        raise NotImplementedError

class SyntheticRegressionData(d2l.DataModule):
    def __init__(self,w,b,noise = 0.01,num_train=1000,num_val=1000,batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train+num_val
        self.X = torch.randn(n,len(w))
        noise = torch.randn(n,1)*noise
        self.y = torch.matmul(self.X,w.reshape((-1,1)))+b+noise

    #This is inefficient with real world data because it requires us to read all the data to start
    def get_dataloader_first_try(self,train):
        #if in training mode
        if train:
            indices = list(range(0,self.num_train))
            # Read the examples in random order
            random.shuffle(indices)
        #if in validation mode we want to read with a fixed order to debug
        else:
            indices = list(range(self.num_trian,self.num_train+self.num_val))
        for i in range(0,len(indices),self.batch_size):
            batch_indices = torch.tensor(indices[i:i+self.batch_size])
            yield self.X[batch_indices],self.y[batch_indices]

    def get_dataloader(self, train):
        #0-num_train is training data while the rest is validation data
        i = slice(0,self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X,self.y),train,i)


data = SyntheticRegressionData(w=torch.tensor([2,-3.4]),b=4.2)

X,y = next(iter(data.train_dataloader()))
print('X shape:',X.shape,'\n y shape:',y.shape)

