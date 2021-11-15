import torch
from torch.nn import init


def get_models(models_config, device, is_data_parallel=False):
    def get_mcd_model_list():
        if "co_detection_cnn" in models_config.model.model_name:
            from module.torch.models.co_detection_cnn import CoDetectionBase, CoDetectionClassifier
            model_g = CoDetectionBase(
                input_ch=models_config.input_channel,
                up_method=models_config.model.up_method,
                up_mode=models_config.model.up_mode,
            )
            model_f1 = CoDetectionClassifier(
                n_class=models_config.output_channel,
                up_method=models_config.model.up_method,
                up_mode=models_config.model.up_mode,
            )
            model_f2 = CoDetectionClassifier(
                n_class=models_config.output_channel,
                up_method=models_config.model.up_method,
                up_mode=models_config.model.up_mode,
            )
        else:
            raise NotImplementedError("Not Found model")

        if models_config.set_init:
            return (model_g.apply(init_weights).to(device),
                    model_f1.apply(init_weights).to(device),
                    model_f2.apply(init_weights).to(device))
        else:
            return model_g.to(device), model_f1.to(device), model_f2.to(device)

    model_list = get_mcd_model_list()

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
    net.apply(init_func)
