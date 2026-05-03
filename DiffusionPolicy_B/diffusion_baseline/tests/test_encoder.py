# 用途: 测试 CNNEncoder 能处理 NHWC uint8/float 图像并输出固定维度表示。
# Purpose: Test CNNEncoder with NHWC uint8/float images and verify representation shape.

import torch

from diffusion_baseline.models.encoder import CNNEncoder


def test_cnn_encoder_uint8_and_float_shapes() -> None:
    repr_dim = 128
    encoder = CNNEncoder(in_channels=3, repr_dim=repr_dim)

    images_uint8 = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    out_uint8 = encoder(images_uint8)
    assert images_uint8.dtype == torch.uint8
    assert out_uint8.dtype == torch.float32
    assert out_uint8.shape == (2, repr_dim)

    images_float = torch.rand(2, 64, 64, 3, dtype=torch.float32)
    out_float = encoder(images_float)
    assert images_float.dtype == torch.float32
    assert out_float.dtype == torch.float32
    assert out_float.shape == (2, repr_dim)
