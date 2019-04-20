#!/usr/bin/env python
# coding: utf-8

# In[7]:


from keras.utils import Sequence
import cv2
import glob,cv2
from keras.callbacks import Callback


# In[5]:


class FlowRgbGenerator(Sequence):
    
    def __init__(self, df,path='/media/brkyzc/veri/20bn-jester-v1/', batch_size=8):
        self.base_path = path 
        self.df = df
        self.batch_size = batch_size

    def __len__(self):
        return int(self.df.count()[0]/self.batch_size)

    def __getitem__(self, cur_batch):
        batch =self.df[cur_batch*self.batch_size : (cur_batch+1)*self.batch_size]
          
          
        batch_inputrgb = []
        batch_inputflow = []
        batch_output = [] 
          
          
        for index,row in batch.iterrows():
            inpflow = get_middle_part_flow(self.base_path + str(row['videoid']) )
            inprgb = get_middle_part_rgb(self.base_path + str(row['videoid']) )
            out = row[1:].values
            
              
            batch_inputflow += [ inpflow ]
            batch_inputrgb += [ inprgb ]
            batch_output += [ out ]

        batch_flowx = np.array(batch_inputflow)
        batch_rgbx = np.array( batch_inputrgb )
        batch_y = np.array( batch_output )

        return  [batch_flowx,batch_rgbx], batch_y 


# In[ ]:


class FlowGenerator(Sequence):
    
    def __init__(self, df,path='/media/brkyzc/veri/20bn-jester-v1/', batch_size=8):
        self.base_path = path 
        self.df = df
        self.batch_size = batch_size

    def __len__(self):
        return int(self.df.count()[0]/self.batch_size)

    def __getitem__(self, cur_batch):
        batch =self.df[cur_batch*self.batch_size : (cur_batch+1)*self.batch_size]
          
          
        batch_inputflow = []
        batch_output = [] 
          
          
        for index,row in batch.iterrows():
            inpflow = get_middle_part_flow(self.base_path + str(row['videoid']) )
            out = row[1:].values
            
            batch_inputflow += [ inpflow ]
            batch_output += [ out ]

        batch_flowx = np.array(batch_inputflow)
        batch_y = np.array( batch_output )
          

        return  batch_flowx, batch_y 


# In[ ]:


class RgbGenerator(Sequence):
    
    def __init__(self, df,path='/media/brkyzc/veri/20bn-jester-v1/', batch_size=8):
        self.base_path = path 
        self.df = df
        self.batch_size = batch_size

    def __len__(self):
        return int(self.df.count()[0]/self.batch_size)

    def __getitem__(self, cur_batch):
        batch =self.df[cur_batch*self.batch_size : (cur_batch+1)*self.batch_size]
          
          
        batch_inputrgb = []
        batch_output = [] 
          
          
        for index,row in batch.iterrows():
            inprgb = get_middle_part_rgb(self.base_path + str(row['videoid']) )
            out = row[1:].values
            
              
            batch_inputrgb += [ inprgb ]
            batch_output += [ out ]

        batch_rgbx = np.array( batch_input )
        batch_y = np.array( batch_output )
          

        return  batch_rgbx, batch_y 


# In[3]:


def read_flow(path):
    with open(path,'rb') as f :
        
        rows= int.from_bytes(f.read(4),byteorder='little')
        cols= int.from_bytes(f.read(4),byteorder='little')
        tp = int.from_bytes(f.read(4),byteorder='little')
        channels =int.from_bytes(f.read(4),byteorder='little')

        data =np.frombuffer(f.read(),np.int8)
        data = np.array([data[::2],data[1::2]])
        data= data.astype(np.float32)
        data =np.reshape(data,(2,rows,cols))
        x =cv2.resize(data[0],(100,150))
        y =cv2.resize(data[1],(100,150))
        xy= np.array([x,y])
        xy= xy.astype(np.int8)

        return np.moveaxis(xy,0,-1)


# In[6]:


def get_middle_part_flow(input_path ):
    flows = sorted(glob.glob(input_path+'/flows/*'))
  
    middle = int(len(flows)/2)
    ims = []
    for ind in range(middle-5,middle+5):
        ims+=[read_flow(flows[ind])]
    data=np.array(ims)
    #print('data shape ' ,data.shape)
    return data


# In[ ]:


def get_middle_part_rgb(input_path ):
    images = sorted(glob.glob(input_path+'/*.jpg'))
  
    middle = int(len(images)/2)
    ims = []
    mx =24
    if len(images)<24:
        mx=len(images)
    for ind in range(mx-20,mx,2):
        ims+=[cv2.resize(cv2.imread(images[ind]),(100,150))]
    #data = np.concatenate(ims,axis=2) #merge multiple frames
    data=np.array(ims)
    data = data/255.
  
    return data
  


# In[ ]:



class PlotLearning(Callback):
    def __init__(self):
        self.i = 0
        self.x = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []


    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)

        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        clear_output(wait=True)
        

        plt.plot(self.x, self.acc, label="accuracy")
        plt.plot(self.x, self.val_acc, label="validation accuracy")
        plt.legend()
        
        plt.show();
        print('vall acc ',self.val_acc[self.i-1],'acc ',self.acc[self.i-1])


