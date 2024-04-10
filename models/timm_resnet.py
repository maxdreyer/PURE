import timm
import torch
import torch.hub
from timm.models import checkpoint_seq
from utils.helper import InspectionLayer
from utils.lrp_canonizers import ResNetCanonizer

def get_resnet18(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_resnet_timm('resnet18.a1_in1k', ckpt_path, pretrained, n_class)


def get_resnet34_timm(ckpt_path=None, pretrained=True, n_class=None) -> torch.nn.Module:
    return get_resnet_timm('resnet34.a1_in1k', ckpt_path, pretrained, n_class)


def get_resnet50_timm(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_resnet_timm('resnet50.a1_in1k', ckpt_path, pretrained, n_class)

def get_resnet101_timm(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_resnet_timm('resnet101.a1_in1k', ckpt_path, pretrained, n_class)

def get_resnet_timm(name, ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    
    model = timm.create_model(name, pretrained, num_classes=n_class) #global_pool=None 11221
    # print(timm.data.resolve_model_data_config(model))
    #model = m(weights=weights)

    if n_class and n_class != 1000:
        num_in = model.fc.in_features
        model.fc = torch.nn.Linear(num_in, n_class, bias=True)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    model.block_0 = InspectionLayer()
    model.block_1 = InspectionLayer()
    model.block_2 = InspectionLayer()
    model.block_3 = InspectionLayer()

    model.forward_features = forward_features_.__get__(model)
    return model

def forward_features_(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
    else:
        x = self.layer1(x)
        x = self.block_0(x)  # added identity
        x = self.layer2(x)
        x = self.block_1(x)  # added identity
        x = self.layer3(x)
        x = self.block_2(x)  # added identity
        x = self.layer4(x)
        x = self.block_3(x)  # added identity
    return x

def forward_(self, x: torch.Tensor) -> torch.Tensor:
    x = self.forward_features(x)
    x = self.forward_head(x)
    x = self.selection(x)
    return x


def get_resnet_canonizer():
    return ResNetCanonizer()
