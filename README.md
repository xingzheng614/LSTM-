#中文文本情感分析(LSTM模型实现)

import paddle
import paddle.fluid as fluid
import numpy as np
import os
from multiprocessing import cpu_count

#公共变量
mydict = {}#字典编码
data_file = 'data/hotel_discuss2.csv' #原始样本文件路径
dict_file = 'data/hotel_dict.txt' #字典文件存放路径
encoding_file = 'data/hotel_encoding.txt'#编码后的样本文件路径
puncts = " \n" #不参与编码的字符
code = 1#生成编码值

######################数据预处理#####################
with open(data_file,'r',encoding='utf-8-sig') as f:
    for line in f.readlines():
        trim_line = line.strip()

        for ch in trim_line:# 拿到每一个字符
            if ch in puncts:
                continue
            if ch in mydict:#字符已经在字典中，编码过了
                continue
            else:
                mydict[ch] = code #分配一个编码,存入到字典
                code += 1
    code += 1
    mydict['<unk>'] = code #未知字符编码

#字典编码已经完成，存入文件中
with open(dict_file,'w',encoding='utf-8-sig') as f:
    f.write(str(mydict))#一行
    print('字典编码保存完成')


#加载字典文件中的内容
def load_dict():
    with open(dict_file,'r',encoding='utf-8-sig') as f:
        lines = f.readlines() #只有一行
        new_dict = eval(lines[0])
    return new_dict

# ----
# 对样本中的评论进行编码

new_dict = load_dict()#加载字典

with open(data_file,'r',encoding='utf-8-sig') as f:
    with open(encoding_file,'w',encoding='utf-8-sig') as fw:
        for line in f.readlines():
            label = line[0] #标签
            remark = line[1:-1] #评论

            for ch in remark: #遍历每一行的文字，进行编码
                if ch in puncts:#如果在不参与编码字符中,则跳过
                    continue
                else:
                    fw.write(str(new_dict[ch])) #写入编码值
                    fw.write(',')
            #for循环结束后，一行编码完成
            fw.write('\t' + str(label) + '\n')

print('数据预处理完成')


#####################模型搭建，训练，保存#####################

#获取字典的长度
def get_dict_len():
    with open(dict_file,'r',encoding='utf-8-sig') as f:
        lines = f.readlines()
        new_dict = eval(lines[0])
    return len(new_dict.keys())


# 创建训练集的reader (打乱顺序)

def data_mapper(sample):
    dt,lbl = sample
    val = [int(word) for word in dt.split(',') if word.isdigit()]
    return val,int(lbl)



def train_reader(train_list_path):
    def reader():
        with open(train_list_path,'r',encoding='utf-8-sig') as f:
            lines = f.readlines()
            np.random.shuffle(lines) #打乱样本数据

            for line in lines:
                data,label = line.split('\t')
                yield data,label

    return paddle.reader.xmap_readers(data_mapper,#二次处理
                                      reader,
                                      cpu_count(),
                                      1024)


#变量
rmk = fluid.layers.data(name='rmk',shape=[1],dtype='int64',lod_level=1)
label = fluid.layers.data(name='label',shape=[1],dtype='int64')

#定义LSTM长短期记忆网络模型
def lstm_net(ipt,input_dim):
    #词嵌入层  :生成词向量
    ipt = fluid.layers.reshape(ipt,[-1,1],inplace=True)
    emb = fluid.layers.embedding(input=ipt,size=[input_dim,128],
                           is_sparse=True) #表示是否为稀疏矩阵数据格式

    #全连接层  size=128
    fc1 = fluid.layers.fc(input=emb,size=128)

    #第一个分支  LSTM +序列池化
    lstm1,_ = fluid.layers.dynamic_lstm(input=fc1,size=128)
    lstm2 = fluid.layers.sequence_pool(input=lstm1,pool_type='max')

    #第二个分支   序列池化
    spool = fluid.layers.sequence_pool(input=fc1,pool_type='max')

    #输出层(分类器)
    out = fluid.layers.fc(input=[lstm2,spool],
                    size=2,
                    act='softmax')

    return out

#获取到字典长度
dict_len = get_dict_len()
#创建模型

model = lstm_net(rmk,dict_len) #model拿到的模型返回的结果

#损失函数,分类任务，使用交叉熵最为损失函数
cost = fluid.layers.cross_entropy(input=model, #预测值
                           label=label) #真实值
avg_cost = fluid.layers.mean(cost)

#准确率
acc = fluid.layers.accuracy(input=model,
                            label=label)

#优化器
optimizer = fluid.optimizer.Adagrad(learning_rate=0.001)
optimizer.minimize(avg_cost)


#执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

reader = train_reader(encoding_file)
batch_train_reader = paddle.batch(reader,batch_size=128)
feeder = fluid.DataFeeder(place=place,feed_list=[rmk,label])


for pass_id in range(5):
    for batch_id,data in enumerate(batch_train_reader()):
        train_cost,train_acc = exe.run(program=fluid.default_main_program(),
                                       feed=feeder.feed(data),
                                       fetch_list=[avg_cost,acc])

        if batch_id % 20 == 0:
            print('pass_id:{},batch_id:{},cost:{},acc:{}'.format(pass_id,
                                                                 batch_id,
                                                                 train_cost[0],
                                                                 train_acc[0]))

print('模型训练完成')

#保存模型
model_save_dir = 'model/chn_emotion/'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

fluid.io.save_inference_model(model_save_dir,
                              feeded_var_names=[rmk.name],
                              target_vars=[model],
                              executor=exe)

print('模型保存完毕')



################ 加载模型 预测 ###############

#加载字典文件中的内容
def load_dict():
    with open(dict_file,'r',encoding='utf-8-sig') as f:
        lines = f.readlines() #只有一行
        new_dict = eval(lines[0])
    return new_dict



#根据字典对　待预测的评论进行编码

def encode_by_dict(remark,dict_encoded):
    remark = remark.strip()
    if len(remark) <= 0:
        return []

    res = []
    for ch in remark:
        if ch in dict_encoded: #文字在字典中
            res.append(dict_encoded[ch])
        else:
            res.append(dict_encoded['<unk>'])

    return res

new_dict = load_dict()
lods = []

lods.append(encode_by_dict("总体来说房间非常干净，卫生也非常不错，交通也很便利",new_dict))
lods.append(encode_by_dict('酒店服务态度极差，我再也不来了',new_dict))
lods.append(encode_by_dict('这个酒店看上去还行，但是太贵了，不建议来',new_dict))
lods.append(encode_by_dict('这个酒店啥都一般，来不来都可以,不推荐也不建议',new_dict))

#获取每个句子的此数量
base_shape = [[len(c) for c in lods]]

#执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
infer_exe.run(fluid.default_startup_program())

#创建lod Tensor
tensor_words = fluid.create_lod_tensor(lods,base_shape,place)

#加载模型
infer_program,feed_target_names,fetch_targets = fluid.io.load_inference_model(dirname=model_save_dir,
                              executor=infer_exe)

#执行预测
results = infer_exe.run(program=infer_program,
                        feed={feed_target_names[0]:tensor_words},
                        fetch_list=fetch_targets)

for i,r in enumerate(results[0]):
    print('负面:{},正面:{}'.format(r[0],r[1]))
