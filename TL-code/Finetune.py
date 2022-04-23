import time

import torch
import torch.nn as nn
from tqdm import tqdm

from Finetune_model import model
from data_preprocess import src_loader, tar_loader

dataloaders = {'src': src_loader,
               'val': tar_loader,
               'tar': tar_loader}
n_epoch = 100
criterion = nn.CrossEntropyLoss()
early_stop = 20


def finetune(model, dataloaders, optimizer):
    since = time.time()
    best_acc = 0
    stop = 0
    for epoch in range(0, n_epoch):
        stop += 1
        # You can uncomment this line for scheduling learning rate
        # lr_schedule(optimizer, epoch)
        for phase in ['src', 'val', 'tar']:
            if phase == 'src':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0, 0
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs, labels = inputs, labels
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'src'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                if phase == 'src':
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct.double() / len(dataloaders[phase].dataset)
            print(f'Epoch: [{epoch:02d}/{n_epoch:02d}]---{phase}, loss: {epoch_loss:.6f}, acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                stop = 0
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'model.pkl')
        if stop >= early_stop:
            break
        print()


    time_pass = time.time() - since
    print(f'Training complete in {time_pass // 60:.0f}m {time_pass % 60:.0f}s')


param_group = []
learning_rate = 0.0001
momentum = 5e-4
for k, v in model.named_parameters():
    if not k.__contains__('fc'):
        param_group += [{'params': v, 'lr': learning_rate}]
    else:
        param_group += [{'params': v, 'lr': learning_rate * 10}]
optimizer = torch.optim.SGD(param_group, momentum=momentum)


finetune(model, dataloaders, optimizer)
print(1)
def test(model, target_test_loader):
    model.eval()
    correct = 0
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data, target
            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = correct.double() / len(target_test_loader.dataset)
    return acc

model.load_state_dict(torch.load('model.pkl'))
acc_test = test(model, dataloaders['tar'])
print(f'Test accuracy: {acc_test}')