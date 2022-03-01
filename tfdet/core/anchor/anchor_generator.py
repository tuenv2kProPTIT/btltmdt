# Copyright (c) OpenMMLab. All rights reserved.
# Modified Tue Nguyen 24/02/2022

from dataclasses import dataclass
from typing import List, Tuple,Union
from xmlrpc.client import Boolean
from tfdet.core.config import CoreConfig
import tensorflow as tf2 
import numpy as np
from tfdet.utils.serializable import keras_serializable

from tfdet.utils.shape_utils import shape_list
@dataclass
class AnchorConfig(CoreConfig):
    """Standard anchor generator for 2D anchor-based detectors.
    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.
    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_priors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_priors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """
    name : str = "AnchorGeneratorConfig"
    last_modified: str = "4:40-24/02/2022"
    #scales (list[int] | None)
    strides: Union[ List[int], List[ Tuple[int, int] ] ] = None 
    ratios : List[float] = None 
    scales : Union[List[int],None] = None 
    base_sizes: Union[List[int], None] =None 
    scale_major: Boolean = True 
    octave_base_scale: int = None 
    scales_per_octave: int = None 
    centers : Union[List[Tuple[float,float]], None] = None 
    center_offset: float = 0. 

@keras_serializable
class AnchorGenerator(tf2.keras.layers.Layer):
    cfg_class=AnchorConfig
    def __init__(self, cfg : AnchorConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg 
        # check center and center_offset
        if self.cfg.center_offset != 0:
            assert self.cfg.centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {self.cfg.centers} is given.'
        if not (0 <= self.cfg.center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{self.cfg.center_offset} is given.')
        if self.cfg.centers is not None:
            assert len(self.cfg.centers) == len(self.cfg.strides), \
                'The number of strides should be the same as centers, got ' \
                f'{self.cfg.strides} and {self.cfg.centers}'

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in  self.cfg.strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if  self.cfg.base_sizes is None else  self.cfg.base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'
        scales=self.cfg.scales
        # calculate scales of anchors
        assert (( self.cfg.octave_base_scale is not None
                 and  self.cfg.scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        
        if scales is not None:
            self.scales =tf2.convert_to_tensor(scales, tf2.float32)
        elif  self.cfg.octave_base_scale is not None and  self.cfg.scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i /  self.cfg.scales_per_octave) for i in range( self.cfg.scales_per_octave)])
            scales = octave_scales *  self.cfg.octave_base_scale
            self.scales =tf2.convert_to_tensor(scales, tf2.float32)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = self.cfg.octave_base_scale
        self.scales_per_octave = self.cfg.scales_per_octave
        self.ratios =tf2.convert_to_tensor(self.cfg.ratios)
        self.scale_major = self.cfg.scale_major
        self.centers = self.cfg.centers
        self.center_offset = self.cfg.center_offset
        self.base_anchors = self.gen_base_anchors()
    

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [base_anchors.shape[0] for base_anchors in self.base_anchors]
    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.
        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors
    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.
        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.
        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = tf2.math.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :])
            hs = (h * h_ratios[:, None] * scales[None, :])
            ws = tf2.reshape(ws,(-1,))
            hs = tf2.reshape(hs,(-1,))
        else:
            ws = (w * scales[:, None] * w_ratios[None, :])
            hs = (h * scales[:, None] * h_ratios[None, :])
            ws = tf2.reshape(ws,(-1,))
            hs = tf2.reshape(hs,(-1,))

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors =tf2.stack(base_anchors, axis=-1)

        return base_anchors
    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.
        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.
        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
       
        if row_major:
            return tf2.meshgrid(x, y)
        else:
            return tf2.meshgrid(y,x)

    def grid_priors(self, featmap_sizes, dtype=tf2.float32):
        """Generate grid anchors in multiple feature levels.
        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            dtype (:obj:`torch.dtype`): Dtype of priors.
                Default: torch.float32.
            device (str): The device where the anchors will be put on.
        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype)
            multi_level_anchors.append(anchors)
        return multi_level_anchors
    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=tf2.float32,
                                 device='cpu',
                                 ):
        """Generate grid anchors of a single level.
        Note:
            This function is usually called by method ``self.grid_priors``.
        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
                ``torch.float32``.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """

        base_anchors = self.base_anchors[level_idx]
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        # with tf2.device(device): Todo in future
        shift_x =tf2.range(0, feat_w,  dtype=dtype)  * stride_w
        shift_y = tf2.range(0, feat_h, dtype=dtype) * stride_h
       

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shift_xx = tf2.reshape(shift_xx,(-1,))
        shift_yy= tf2.reshape(shift_yy,(-1,))
        shifts = tf2.stack([shift_yy,shift_xx, shift_yy,shift_xx], axis=-1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors =  base_anchors[None, :, :] + shifts[:, None, :]  
        all_anchors =tf2.reshape(all_anchors, (-1, 4))
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors
    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=tf2.float32,
                      device='cuda'):
        """Generate sparse anchors according to the ``prior_idxs``.
        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (h, w).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 4), N should be equal to
                the length of ``prior_idxs``.
        """

        height, width = featmap_size
        num_base_anchors = self.num_base_anchors[level_idx]
        base_anchor_id = prior_idxs % num_base_anchors
        x = (prior_idxs //
             num_base_anchors) % width * self.strides[level_idx][0]
        y = (prior_idxs // width //
             num_base_anchors) % height * self.strides[level_idx][1]

        priors = tf2.stack([x, y, x, y],1) + self.base_anchors[level_idx][base_anchor_id,:]

    

        return priors
    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.
        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.
        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags
    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        """Generate the valid flags of anchor in a single feature map.
        Args:
            featmap_size (tuple[int]): The size of feature maps.
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.
        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
       
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = tf2.where(tf2.range(feat_w) < valid_w,1,0)
        valid_y =tf2.where(tf2.range(feat_h) < valid_h,1,0)
        
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid_xx = tf2.reshape(valid_xx,[-1,])
        valid_yy= tf2.reshape(valid_yy,[-1,])
        
        valid =tf2.where(tf2.logical_and(valid_xx==1, valid_yy==1),1,0) #tf.math.logical_and( valid_xx , valid_yy)
        valid= tf2.expand_dims(valid, axis=-1) 
        
        valid =tf2.broadcast_to(valid, [valid.shape[0],num_base_anchors])
        valid = tf2.reshape(valid,[-1,])
        return valid

    def call(self, inputs):
        ''' inputs : bs,h,w,channel 
        '''
        shape_ = shape_list(inputs)
        all_anchor = self.grid_priors(shape_[1:-1])
        return all_anchor

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str

def _pair(value):
    if isinstance(value,list):
        assert len(value) == 2
        return tuple(value)
    if isinstance(value, tuple):
        assert len(value) == 2
        return value
    return (value,value)