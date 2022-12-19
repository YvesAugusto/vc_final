from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch.nn import functional as F


class Intercept:

    def __init__(self, model) -> None:
        self.model: InceptionResnetV1 = model

    def forward(self, x):
        outputs = {}

        x = self.model.conv2d_1a(x)
        outputs['conv2d_1a'] = x

        x = self.model.conv2d_2a(x)
        outputs['conv2d_2a'] = x

        x = self.model.conv2d_2b(x)
        outputs['conv2d_2b'] = x

        x = self.model.maxpool_3a(x)
        outputs['maxpool_3a'] = x

        x = self.model.conv2d_3b(x)
        outputs['conv2d_3b'] = x

        x = self.model.conv2d_4a(x)
        outputs['conv2d_4a'] = x

        x = self.model.conv2d_4b(x)
        outputs['conv2d_4b'] = x

        x = self.model.repeat_1(x)
        outputs['repeat_1'] = x

        x = self.model.mixed_6a(x)
        outputs['mixed_6a'] = x

        x = self.model.repeat_2(x)
        outputs['repeat_2'] = x

        x = self.model.mixed_7a(x)
        outputs['mixed_7a'] = x

        x = self.model.repeat_3(x)
        outputs['repeat_3'] = x

        x = self.model.block8(x)
        outputs['block8'] = x

        x = self.model.avgpool_1a(x)
        outputs['avgpool_1a'] = x

        x = self.model.dropout(x)
        outputs['dropout'] = x

        x = self.model.last_linear(x.view(x.shape[0], -1))
        outputs['last_linear'] = x

        x = self.model.last_bn(x)
        outputs['last_bn'] = x

        x = F.normalize(x, p=2, dim=1)
        outputs['last_bn_norm'] = x

        return outputs
