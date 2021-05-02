#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets 
import PIL
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[16]:


os.getcwd() #getting the path to check where we 


# In[17]:


#checking for gpu support

if(torch.cuda.is_available()):
    print("gpu support available and shifting to the same")
    device=torch.device("cuda:0")
    print(device)
else:
    print("no gpu support available hence would be using cpu")


# In[18]:


training_data_path=r"D:\Achema\02_Challenge by TU Dortmund\Training2\Training"  #getting the path for training
val_data_path=r"D:\Achema\02_Challenge by TU Dortmund\Validation" #getting the path for validation

#checked the image quality on reducing the size of the image in pixels and seems to look good..checked in paint 
transformation = transforms.Compose([transforms.Resize((500,500)), #the resolution is staying nice when checked with paint hence made the image smaller
    transforms.ToTensor()])

train_dataset=datasets.ImageFolder(training_data_path,transform=transformation)
val_dataset=datasets.ImageFolder(val_data_path,transform=transformation)


# In[19]:


#making the Dataloader...as the images are already segmented into descrete folders with their labels as folder names no need 
#to create a custom dataloader

train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_dataloader=DataLoader(val_dataset,batch_size=32,shuffle=False) #true should also work but not necessary


# In[20]:


# Printing a random image from the data 

image, label = next(iter(train_dataloader))


# In[21]:


print(image.size())

plt.imshow(image[0].permute(1, 2, 0))


# In[22]:


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,10,5)
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2=nn.Conv2d(10,20,5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1=nn.Linear(20*122*122,100)
        self.fc2=nn.Linear(100,50)
        self.fc3=nn.Linear(50,2)
        
    def forward(self,x):
        val=self.avgpool1(F.relu(self.conv1(x)))
        val=self.maxpool2(F.relu(self.conv2(val)))
        val=val.view(-1,20*122*122)
        val=F.relu(self.fc1(val))
        val=F.relu(self.fc2(val))
        val=F.sigmoid(self.fc3(val))
        return val

    
net=Net()


# In[23]:


# print(net)
net.to(device)


# In[24]:


epochs=7
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)


# In[25]:


print(len(train_dataset))


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


# acc_list = []
for i in range(epochs):
    correct=0
    for values in train_dataloader:

        data,label =values
        data,label =data.to(device), label.to(device) #passing data and labels to the gpu along with our model 
        net.zero_grad()
        output=net(data)
        output = torch.round(output.squeeze())

        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        
#     #accuracy check :
        size = label.size(0)
#         print(size)
        _, model_prediction= torch.max(output.data, 1)
        correct = (model_prediction == label).type(torch.float).sum().item()
    check_acc=correct/size
    
    print("Epoch {}/{}, Loss: {:.3f}".format(i+1,epochs, loss.item()))
    print("Accuracy: {:.3f}".format(check_acc))
#     print(loss)
print("training is done")


# In[29]:


# # acc_list = []
# for i in range(epochs):
#     correct=0
#     for values in val_dataloader:

#         data,label =values
#         data,label =data.to(device), label.to(device) #passing data and labels to the gpu along with our model 
#         net.zero_grad()
#         output=net(data)
#         output = torch.round(output.squeeze())

#         loss=criterion(output,label)
#         loss.backward()
#         optimizer.step()
        
# #     #accuracy check :
#         size = label.size(0)
# #         print(size)
#         _, model_prediction= torch.max(output.data, 1)
#         correct = (model_prediction == label).type(torch.float).sum().item()
#     check_acc=correct/size
    
#     print("Epoch {}/{}, Loss: {:.3f}".format(i+1,epochs, loss.item()))
#     print("Accuracy: {:.3f}".format(check_acc))
# #     print(loss)


# In[ ]:





# In[ ]:




