from mxnet import autograd, nd, init
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet import gluon


# 生成数据
X_num = 2 # 特征数为2
X_samples = 1000 #样本数
true_w =nd.array([2, -3.4])
true_b = 4.2
features = nd.random.normal(scale=1,shape=(X_samples, X_num))
labels = nd.dot(features, true_w.T) + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

batch_size = 10

# 获取数据
dataset = gdata.ArrayDataset(features,labels)

data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)

for X,y in data_iter:
    print(X,y)
    break


# 创建模型
net = nn.Sequential()
net.add(nn.Dense(1))

# 初始化模型参数 均值为0, 标准差为0.01
net.initialize(init.Normal(sigma=0.01))

# 定义损失函
loss = gloss.L2Loss() #平方损失 L2范数损失

# 定义优化算法
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

# 训练模型
epoches = 10

for epoch in range(1,epoches+1):
    for X,y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features),labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

# 检验效果
dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())

