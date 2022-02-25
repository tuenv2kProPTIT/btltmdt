"""
Convolution with activation and norm
"""
from typing import Tuple, Union

import tensorflow as tf

from tfdet.utils.etc import to_2tuple
from tfdet.models.layers.factory import act_layer_factory, norm_layer_factory
def get_padding(
    kernel_size: Union[int, tuple],
    strides: Union[int, tuple] = 1,
    dilation_rate: Union[int, tuple] = 1,
) -> Tuple[int, int]:
    """Calculate symmetric padding for a convolution"""
    kernel_size = to_2tuple(kernel_size)
    strides = to_2tuple(strides)
    dilation_rate = to_2tuple(dilation_rate)
    padding = (
        ((strides[0] - 1) + dilation_rate[0] * (kernel_size[0] - 1)) // 2,
        ((strides[1] - 1) + dilation_rate[1] * (kernel_size[1] - 1)) // 2,
    )
    return padding


class Conv2DNorm(tf.keras.layers.Conv2D):
    """
    Conv2D with Weight Standardization. Used for BiT ResNet-V2 models.
    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight
        Standardization`
    Link: https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        act_cfg=None,
        norm_cfg=None,
        order = ('conv','norm','act'),
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        eps=1e-8,
        **kwargs
    ):
        if norm_cfg is not None and norm_cfg.get("name","") != "":
            use_bias=False
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding if padding != "symmetric" else "valid",
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.act_cfg=act_cfg if act_cfg else {"name":"linear"}
        self.norm_cfg=norm_cfg if norm_cfg else {"name":""}
        self.order=order
        self.eps = eps
        self.pad = (
            tf.keras.layers.ZeroPadding2D(
                padding=get_padding(kernel_size, strides, dilation_rate)
            )
            if padding == "symmetric"
            else None
        )
        norm_fc = norm_layer_factory(self.norm_cfg.get("name"))
        norm_cfg = {k:v for k,v in self.norm_cfg.items()}
        norm_cfg.pop("name","")
        self.norm = norm_fc(**norm_cfg)

        act_fc = act_layer_factory(self.act_cfg.get("name"))
        act_cfg={k:v for k,v in self.act_cfg.items()}
        act_cfg.pop("name","")
        self.act = act_fc(**act_cfg)
    def get_config(self):
        cfg = super().get_config()
        cfg.update(act_fg=self.act_cfg, norm_cfg=self.norm_cfg,order=self.order)
        return cfg 
    def call(self, x, training=None):
        if self.pad is not None:
            x = self.pad(x)
        
        convx = super().call(x)
        if self.order[-1] == 'act':
            convx = self.norm(convx, training=training)
            convx = self.act(convx)
        else:
            convx = self.act(convx)
            convx = self.act(convx, training=training)
        return convx