import torch
from torch.nn import init


def get_models(net_name, input_ch, n_class, device, res="50", method="MCD", uses_one_classifier=False,
               is_data_parallel=False, up_mode='upsample', junction_point=0):
    def get_mcd_model_list():
        if net_name == "fcn":
            from module.torch.models.fcn import ResBase, ResClassifier
            model_g = ResBase(n_class, layer=res, input_ch=input_ch)
            model_f1 = ResClassifier(n_class)
            model_f2 = ResClassifier(n_class)
        elif net_name == "fcnvgg":
            from module.torch.models.vgg_fcn import FCN8sBase, FCN8sClassifier
            model_g = FCN8sBase(n_class)
            model_f1 = FCN8sClassifier(n_class)
            model_f2 = FCN8sClassifier(n_class)
        elif "drn" in net_name:
            from module.torch.models.dilated_fcn import DRNSegBase, DRNSegPixelClassifier_ADR
            if uses_one_classifier:
                model_g = DRNSegBase(model_name=net_name, n_class=n_class, input_ch=input_ch)
                model_f1 = DRNSegPixelClassifier_ADR(n_class=n_class)
                model_f2 = DRNSegPixelClassifier_ADR(n_class=n_class)
            else:
                from module.torch.models.dilated_fcn import DRNSegBase, DRNSegPixelClassifier
                model_g = DRNSegBase(model_name=net_name, n_class=n_class, input_ch=input_ch)
                model_f1 = DRNSegPixelClassifier(n_class=n_class)
                model_f2 = DRNSegPixelClassifier(n_class=n_class)
        elif "unet" in net_name:
            from module.torch.models.unet import UNetBase, UNetClassifier
            model_g = UNetBase(in_channels=input_ch, up_mode=up_mode, junction_point=junction_point, factor=1)
            model_f1 = UNetClassifier(n_classes=n_class, up_mode=up_mode, junction_point=junction_point, factor=1)
            model_f2 = UNetClassifier(n_classes=n_class, up_mode=up_mode, junction_point=junction_point, factor=1)
        elif "co_detection_cnn" in net_name:
            from module.torch.models.co_detection_cnn import CoDetectionBase, CoDetectionClassifier
            model_g = CoDetectionBase(n_channels=input_ch, up_mode=up_mode)
            model_f1 = CoDetectionClassifier(n_classes=n_class, up_mode=up_mode)
            model_f2 = CoDetectionClassifier(n_classes=n_class, up_mode=up_mode)
        elif "co_detection_cnn_j1" in net_name:
            from module.torch.models.co_detection_cnn import CoDetectionBaseJ1, CoDetectionClassifier
            model_g = CoDetectionBaseJ1(n_channels=input_ch, up_mode=up_mode)
            model_f1 = CoDetectionClassifier(n_classes=n_class, up_mode=up_mode)
            model_f2 = CoDetectionClassifier(n_classes=n_class, up_mode=up_mode)
        else:
            raise NotImplementedError("Only FCN (Including Dilated FCN), SegNet, PSPNetare supported!")

        return model_g.apply(init_weights).to(device), model_f1.apply(init_weights).to(device), model_f2.apply(
            init_weights).to(device)

    if method == "MCD":
        model_list = get_mcd_model_list()
    else:
        return NotImplementedError("Sorry... Only MCD is supported!")

    if is_data_parallel:
        return [torch.nn.DataParallel(x) for x in model_list]
    else:
        return model_list


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
