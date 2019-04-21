#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


from models import ModelType,ModelFactory


# In[3]:


model =ModelFactory(rgbpath='trained_models/rgblstm.h5',trained=True).getModel(ModelType.RGB)


# In[4]:


model.summary()


# In[5]:


from keras.layers import Input
from keras.models import Model


# In[6]:


rgbinput = Input((150,100,3))

x = model.layers[1].layer(rgbinput)
for layer in model.layers[2:-3]:
    x = layer.layer(x)
x


# In[7]:


encoder = Model(inputs = rgbinput , outputs =x)
encoder.summary()


# In[8]:


lstminput = Input((10,1024))

x = model.layers[-2](lstminput)
x = model.layers[-1](x)
x


# In[9]:


lstm = Model(inputs = lstminput , outputs =x)
lstm.summary()


# In[10]:


import pandas as pd
valid = pd.read_csv('20bnjester_csv_files/valid.csv')


# In[ ]:


from collections import deque
import numpy as np
import time
q = deque([np.zeros(1024) for i in range(10)] )#queue of extracted features , initialy filled with zeros 
def grab_frame(cap):
    ret,frame = cap.read()
    return frame


import matplotlib.animation as animation
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec


gs = gridspec.GridSpec(2, 3)  #2 rows  3 cols

fig = pl.figure()

ax1 = pl.subplot(gs[0, 1]) 
ax2 = pl.subplot(gs[1, :]) 



cap = cv2.VideoCapture(0)

graph,= ax2.plot(valid.columns.values[1:] ,np.arange(27))

im1 = ax1.imshow(grab_frame(cap))

ax2.set_ylim(0, 1)

def animate(i):
    
    ret,frame = cap.read()
    
    im1.set_data(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    q.popleft()
    plt.setp(ax2.get_xticklabels(), rotation=20, horizontalalignment='right')
    q.append(encoder.predict(np.array([cv2.resize(frame/255.,(100,150))]))[0])
    graph.set_data(valid.columns.values[1:],lstm.predict(np.array([q]))[0])
    
    return im1,graph,
    


ani = animation.FuncAnimation(fig,animate,blit=True,repeat=True,interval=5)
plt.show()


