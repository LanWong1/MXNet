
# coding: utf-8

# # 线性回归

# ### 导入包

# In[ ]:



from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random


# ### 生成数据

# In[ ]:


num_x = 2
num_samples = 1000
true_W =nd.array([5,-5.8]) 
true_b = 3.4

X = nd.random.normal(scale=1,shape=(num_samples,num_x))
lables = nd.dot(X,true_W.T)+ true_b
lables += nd.random.normal(scale=0.01,shape=lables.shape) 


def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(5,5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
set_figsize()
#plt.scatter(X[:,1].asnumpy(),lables.asnumpy())


# ### 读取数据




#mini_batch

def data_itr(x,lables,batch_size):
    num_samples = len(lables)
    index = list(range(num_samples))
    random.shuffle(index)#随机index 
    for i in range(0, num_samples,batch_size):
       
        j = nd.array(index[i:min(i+batch_size,num_samples)])#按顺序取得数据
        yield x.take(j),lables.take(j)
        
        


# In[34]:


batch_size = 10
for x,y in data_itr(X,lables,batch_size):
    print(x,y)
    break
 


# ### 初始化参数

# In[35]:


w = nd.random.normal(scale=0.01,shape=(num_x,1))
b = nd.zeros(shape=(1,))


# In[36]:


w.attach_grad()
b.attach_grad()


# ### 定义模型

# In[37]:


def liner_regre(x,w,b):
    return nd.dot(x,w)+b


# ### 定义损失函数

# In[38]:


def squared_loss(y_hat,y):
    m = 2*len(y)
   
    return (y_hat-y.reshape(y_hat.shape))**2/m


# ### 定义优化算法

# In[39]:


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr*param.grad/batch_size


# ### 训练模型

# In[41]:


lr = 0.03
epoches = 50
net = liner_regre
loss = squared_loss
loss_his = []
for epoch in range(epoches):
    
    for x,y in data_itr(X,lables,batch_size):
        
        with autograd.record():
            l = loss(net(x,w,b),y)
        l.backward()
        sgd([w,b],lr,batch_size)
        
    train_l = loss(net(X,w,b),lables)
    loss_his.append(train_l.mean().asnumpy())
    print('epoche %d loss %f '''% (epoch+1,train_l.mean().asnumpy()))
    

epo = range(epoches)
print(epo)
print(loss_his)
plt.plot(epo,loss_his)
plt.show()




