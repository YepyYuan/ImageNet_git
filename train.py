import torch
import torch.utils
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from datetime import datetime
import os


def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, num_epoches):
    cur_datetime = datetime.now()
    date_str = '{:0>4d}{:0>2d}{:0>2d}'.format(cur_datetime.year, cur_datetime.month, cur_datetime.day)
    time_str = cur_datetime.strftime('%H%M%S')

    model_save_path = './models/' + date_str + '_' + time_str
    if os.path.exists(model_save_path) == False:
        os.makedirs(model_save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter('runs/'+ date_str + '_' + time_str)

    best_test_acc = 0

    for epoch in range(num_epoches):

        itrs = 0

        train_loss_sum = 0.0
        train_acc = 0.0
        correct = 0
        total = 0

        ## train
        model.train()

        for data in train_dataloader:
            images, labels = data
            images, labels = Variable(images.to(device)), Variable(labels.to(device))

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            _,prediction = torch.max(output, dim=1)

            total += labels.size(0)
            correct += prediction.eq(labels).cpu().sum()
            itrs += 1

            train_loss_avg = train_loss_sum / itrs
            train_acc = correct / total
        
        model_weights = model.state_dict()
        if (epoch+1) % 5 == 0:
            torch.save(model_weights, model_save_path + '/model_epoch_{:0>3d}.pth'.format(epoch+1))

        print('Epoch %d| %d training complete!' %(epoch+1, num_epoches))
        print('-'*30)
        print('train loss: %.6f acc: %.3f' %(train_loss_avg, train_acc))

        test_correct = 0
        test_total = 0
        test_loss = []

        ## test
        model.eval()

        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            loss = criterion(output, labels)

            _,prediction = torch.max(output, dim=1)

            test_total += labels.size(0)
            test_correct += prediction.eq(labels).cpu().sum()
            test_acc = test_correct / test_total
            test_loss.append(loss.item())
        
        test_loss = sum(test_loss)/len(test_loss)

        if test_acc > best_test_acc:
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, model_save_path + '/best_model.pth')


        writer.add_scalars('loss', {'train': train_loss_avg, 'test':test_loss} , epoch)
        writer.add_scalars('acc', {'train': train_acc, 'test':test_acc}, epoch)
        
        print('test loss: %.6f acc: %.3f' %(test_loss, test_acc))

    writer.close()