from torchvision.models import resnet18
import torch
import torch.nn as nn


def load_resnet18(pretrained = False, weights_dir = None):

    if weights_dir:
        model = resnet18()
        model_dict = model.state_dict()

        pretrained_model = torch.load(weights_dir)
        pretrained_dict = pretrained_model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    else:
        model = resnet18(pretrained = pretrained)

    # change the 'fc' layer for our tast

    fc_infeatures = model.fc.in_features
    model.fc = nn.Linear(fc_infeatures,200)

    return model