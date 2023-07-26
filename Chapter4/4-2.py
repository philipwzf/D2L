import time
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l
from matplotlib import pyplot as plt

class FashionMNIST(d2l.DataModule):
    def __init__(self,batch_size=64,resize=(28,28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(root=self.root,train=True,transform=trans,download=True)
        self.val = torchvision.datasets.FashionMNIST(root=self.root,train=False,transform=trans,download=True)

    def text_labels(self,indices):
        """Return text labels"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]
    
    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data,self.batch_size,shuffle=train,num_workers=self.num_workers)
    
    def visualize(self,batch,nrows=1,ncols=8,labels=[]):
        X,y=batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1),nrows,ncols,titles=labels)


def show_images(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    """Plot a list of images"""
    figsize = (num_cols*scale,num_rows*scale)
    _, axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    for i, (ax,img) in enumerate(zip(axes,imgs)):
        try:
            img = img.detach().numpy()
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

    
if __name__ == '__main__':
    data = FashionMNIST(resize=(32,32))
    #X,y = next(iter(data.train_dataloader()))
    #print(X.shape,X.dtype,y.shape,y.dtype)
    #tic = time.time()
    #for X,y in data.train_dataloader():
        #continue
    #print(f'{time.time()-tic:.2f} sec')

    batch = next(iter(data.val_dataloader()))
    data.visualize(batch)
    plt.show()

