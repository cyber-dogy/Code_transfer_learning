from torch import nn
#from data_test import train_file,test_file
from classifier_pre import train_file,test_file
from classifier_model import model, optimizer, loss_f
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

#%%参数设置
BATCH_SIZE = 1
EPOCH = 50
m = nn.Sigmoid()
#%% tensorboard
writer = SummaryWriter('./logs/')
#%% 制作数据加载器

train_loader = DataLoader(
    dataset=train_file,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_file,
    batch_size=BATCH_SIZE,
    shuffle=True
)

#%% 定义计算整个训练集或测试集loss及acc的函数
def calc(data_loader):
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.type(torch.FloatTensor)
            targets = targets
            output = model(data)
            loss += loss_f(m(output), targets.float())
            correct += (output.argmax(1) == targets.argmax(1)).sum()
            total += data.size(0)
    loss = loss.item()/len(data_loader)
    acc = correct.item()/total
    return loss, acc


#%% 训练过程打印函数
def show():
    # 定义全局变量
    if epoch == 0:
        global model_saved_list
        global temp
        temp = 0
    # 打印训练的EPOCH和STEP信息

    header_list = [
        f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
        f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}'
    ]
    header_show = ' '.join(header_list)
    #print(header_show, end=' ')
    # 打印训练的LOSS和ACC信息
    loss, acc = calc(train_loader)
    writer.add_scalar('loss', loss, epoch+1)
    writer.add_scalar('acc', acc, epoch+1)

    train_list = ['\n',
        f'LOSS: {loss:.4f}',
        f'ACC: {acc:.4f}'
    ]
    train_show = ' '.join(train_list)
    print(train_show, end=' ')
    # 打印测试的LOSS和ACC信息
    val_loss, val_acc = calc(test_loader)
    writer.add_scalar('val_loss', val_loss, epoch+1)
    writer.add_scalar('val_acc', val_acc, epoch+1)
    test_list = [
        f'VAL-LOSS: {val_loss:.4f}',
        f'VAL-ACC: {val_acc:.4f}'
    ]
    test_show = ' '.join(test_list)
    print(test_show, end=' ')
    # 保存最佳模型
    if val_acc > temp:
        model_saved_list = header_list+train_list+test_list
        torch.save(model.state_dict(), 'model.pt')
        temp = val_acc
#%%训练模型

for epoch in tqdm(range(EPOCH)):
    model.train()
    time.sleep(0.05)
    start_time = time.time()
    for step, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.type(torch.FloatTensor)
        out = model(data)
        loss = loss_f(m(out), label.float())
        acc = (out.argmax(1) == label.argmax(1)).sum().item() / BATCH_SIZE
        loss.backward()
        optimizer.step()
        #print(f'LOSS: {loss.item():.4f}',f'ACC: {acc:.4f}')
    show()
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time-start_time)}')

#%% 打印并保存最优模型的信息
model_saved_show = ' '.join(model_saved_list)
print('| BEST-MODEL | '+model_saved_show)
with open('model_1d.txt', 'a') as f:
    f.write(model_saved_show+'\n')
#%% tensorboard
'''
tensorboard --logdir=网络/fault_classification/logs --port 9000
'''






