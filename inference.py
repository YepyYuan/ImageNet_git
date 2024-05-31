from res18 import load_resnet18
import torch

def inference(model_weights_path, test_data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_resnet18(pretrained= False).to(device)
    model_weights = torch.load(model_weights_path)
    model.load_state_dict(model_weights)

    model.eval()

    test_total = 0
    test_correct = 0

    for data in test_data:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        _,prediction = torch.max(output, dim=1)

        test_total += labels.size(0)
        test_correct += prediction.eq(labels).cpu().sum()
        
    test_acc = test_correct / test_total

    return test_acc
