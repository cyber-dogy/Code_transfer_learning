import torch
import torchvision.models as models
path=r'D:\故障特征提取\python路径\GAN-实践\generator.pth'
net=torch.load(path)
print(net)

# pretrained=True就可以使用预训练的模型
#net1 = models.squeezenet1_1(pretrained=False)
#net1.load_state_dict(torch.load(path))
#print(net1)