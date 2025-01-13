import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from .densenet import DenseNet3
from .wideresnet import WideResNet

class EnsembleNet(torch.nn.Module):
    def __init__(self, teacher_names, exp_cat_name ='vanilla', teacher_epochs=100, model_arch = 'wideresnet', num_classes=10, normalizer=None, in_dataset='CIFAR-10'):
        super(EnsembleNet, self).__init__()
        self.ensemble_size = len(teacher_names.strip().split(','))
        self.output_size = num_classes

        self.in_dataset = in_dataset
        self.teacher_epochs = teacher_epochs
        self.teacher_names = teacher_names
        self.model_arch = model_arch
        
        ensemble_models = []
        for t_name in teacher_names.strip().split(','):
            if model_arch == 'densenet':
                t_model = DenseNet3(100, num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=normalizer)
            elif model_arch == 'wideresnet':
                t_model = WideResNet(40, num_classes, widen_factor=2, dropRate=0.0, normalizer=normalizer)              
            else:
                assert False, 'Not supported model arch: {}'.format(model_arch)

            checkpoint = torch.load("./checkpoints/{exp_cat_name}/{in_dataset}/no_auxiliary/{model_arch}/{name}/checkpoint_{epochs}.pth.tar"
                                .format(exp_cat_name=exp_cat_name, in_dataset=in_dataset, model_arch=model_arch, name=t_name, epochs=teacher_epochs))                         
            t_model.load_state_dict(checkpoint['state_dict'])
            ensemble_models.append(t_model)
        self.ensemble_models = nn.ModuleList(ensemble_models)

    def forward(self, x):
        output_list = []
        for sub_model in self.ensemble_models:
            output_list.append(sub_model(x))
        ensemble_outputs = torch.zeros(output_list[0].shape).cuda()
        for out in output_list:
            ensemble_outputs += 1/len(output_list) * out        
        return ensemble_outputs
    
    def get_ensemble_outputs(self, x):
        batch_size = x.size(0)
        logits = torch.zeros((batch_size, self.ensemble_size, self.output_size)).cuda()
        for sub_model_ind, sub_model in enumerate(self.ensemble_models):
            logits[:, sub_model_ind, :] = sub_model(x)        
        return logits
