from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict
import timm


def create_model(num_classes=1, image_size=512, architecture="tf_efficientnet_d0"):

    if architecture == "tf_efficientnetv2_l":
        efficientdet_model_param_dict['tf_efficientnetv2_l'] = dict(
            name='tf_efficientnetv2_l',
            backbone_name='tf_efficientnetv2_l',
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=num_classes,
            url='', )

    elif architecture == "tf_efficientnet_d0":
        efficientdet_model_param_dict['tf_efficientnet_d0'] = dict(
            name='tf_efficientnet_d0',
            backbone_name='tf_efficientnet_d0',
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=num_classes,
            url='', )
    else:
        raise ValueError('Update architecture configuration here!')

    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)


if __name__ == '__main__':
    # print(f'number of configs: {len(efficientdet_model_param_dict)}')
    print(list(efficientdet_model_param_dict.keys())[::3])
    print(timm.list_models('tf_efficientnetv2_*'))
    # model = create_model()