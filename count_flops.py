from fvcore.nn import FlopCountAnalysis
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from faster_rcnn import FasterRCNNModel

import torch.nn as nn

import torch


def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    # efficientdet_model_param_dict["tf_efficientnetv2_l"] = dict(
    #     name="tf_efficientnetv2_l",
    #     backbone_name="tf_efficientnetv2_l",
    #     backbone_args=dict(drop_path_rate=0.2),
    #     num_classes=num_classes,
    #     url="",
    # )

    config = get_efficientdet_config(architecture)
    config.update({"num_classes": num_classes})
    config.update({"image_size": (image_size, image_size)})

    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = nn.Identity()
    net.box_net = nn.Identity()
    # net.load_state_dict(
    #     torch.load("results/weights/barlow_pretrained_weights_backbone")
    # )
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    net.box_net = HeadNet(config, num_outputs=4)
    return net


# model = create_model(image_size=640, architecture="tf_efficientdet_d1")
model = FasterRCNNModel(num_classes=2, batch_size=1, num_workers=1)
model.eval()

inputs = torch.randn((1, 3, 512, 512), dtype=torch.float32)

flops = FlopCountAnalysis(model, inputs)
print(flops.total())
