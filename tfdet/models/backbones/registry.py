"""
Model Registry
Based on timm/models/registry.py  by Ross Wightman
Copyright 2020 Ross Wightman
Copyright 2021 Martins Bruveris
Copyright 2022 TueNguyen
Modified by Tue Nguyen 24/02/2022
"""

from ast import mod
import fnmatch
import re
import sys
from collections import defaultdict
from copy import deepcopy
from typing import List, Union
from tfdet.utils.comvert import load_pytorch_weights_in_tf2_model
__all__ = [
    "list_backbones",
    "is_backbones",
    "is_backbones_in_modules",
    "list_modules",
    "backbones_class",
    "backbones_config",
    "register_backbone",
    "get_backbone"
    
]

_backbones_class = {}
_backbones_config = {}
# Dict of sets to check membership of model in module
_module_to_models = defaultdict(set)
_model_has_pretrained = set()  # Model names that have pretrained weight url present

_backbone_class_sum={}
def register(cls):
    cfg = cls.cfg_class
    name=cfg.name.lower()
    _backbone_class_sum[name] = {
        'config':cfg,
        'instance':cls
    }
    return cls
def get_backbone(cfg):
    name=cfg.get("name","")
    if name in list_backbones():
        instance= backbones_class(name)(backbones_config(name)(**cfg))
        if 'url' in cfg:
            if cfg['url'] == 'timm':
                try:
                    import timm
                    model_torch = timm.create_model(name, pretrained=True)
                    weight_torch=model_torch.state_dict() 
                    load_pytorch_weights_in_tf2_model(instance, weight_torch)
                except Exception as e:
                    print(e) 
            else:
                # support pretrained weights tensorflow
                instance.load_weights(cfg['url'])
        return instance
    name = name.lower()
    if name in _backbone_class_sum:
        instance= _backbone_class_sum[name]['instance'](_backbone_class_sum[name]['config'](**cfg))
        if 'url' in cfg:
            if cfg['url'] == 'timm':
                try:
                    import timm
                    model_torch = timm.create_model(name, pretrained=True)
                    weight_torch=model_torch.state_dict() 
                    load_pytorch_weights_in_tf2_model(instance, weight_torch)
                except Exception as e:
                    print(e) 
            else:
                # support pretrained weights tensorflow
                instance.load_weights(cfg['url'])
        return instance
    raise Exception(f"class with name = {name} didn't register at anywhere")
def register_backbone(fn):
    # Get model class and model config
    cls, cfg = fn()
    model_name = cfg.name
    if fn.__name__ != model_name:
        raise ValueError(f"Model name({model_name}) != function name ({fn.__name__}).")

    # Lookup module, where model is defined
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split(".")
    module_name = module_name_split[-1] if len(module_name_split) else ""

    # Add model function to __all__ in that module
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # Add entries to registry dict/sets
    _backbones_class[model_name] = cls
    _backbones_config[model_name] = deepcopy(cfg)
    _module_to_models[module_name].add(model_name)
    if cfg.url:  # If URL is non-null, we assume it points to pretrained weights
        _model_has_pretrained.add(model_name)

    return fn

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def list_backbones(
    name_filter: Union[str, List[str]] = "",
    module: str = "",
    pretrained: Union[bool, str] = False,
    exclude_filters: Union[str, List[str]] = "",
):
    """Returns list of available model names, sorted alphabetically.
    Args:
        name_filter: Wildcard filter string that works with fnmatch
        module: Limit model selection to a specific sub-module (ie "resnet")
        pretrained: If True only include models with pretrained weights. If "timm",
            only include models with pretrained weights in timm library
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after
            including them with filter
    Example:
        model_list("gluon_resnet*") -- returns all models starting with "gluon_resnet"
        model_list("*resnext*", "resnet") -- returns all models with "resnext" in
            "resnet" module
    """
    if module:
        all_models = list(_module_to_models[module])
    else:
        all_models = _backbones_class.keys()

    if name_filter:
        if not isinstance(name_filter, (tuple, list)):
            name_filter = [name_filter]
        models = set()
        for f in name_filter:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = models.union(include_models)
    else:
        models = set(all_models)

    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = models.difference(exclude_models)

    if pretrained is True:
        models = _model_has_pretrained.intersection(models)
    elif pretrained == "timm":
        models = models.intersection(_timm_pretrained_models())

    return list(sorted(models, key=_natural_key))


def is_backbones(model_name):
    """Check if a model name exists"""
    return model_name in _backbones_class


def backbones_class(model_name):
    """Fetch a model entrypoint for specified model name"""
    return _backbones_class[model_name]


def backbones_config(model_name):
    """Fetch a model config for specified model name"""
    return _backbones_config[model_name]


def list_modules():
    """Return list of module names that contain models / model entrypoints"""
    modules = _module_to_models.keys()
    return list(sorted(modules))


def is_backbones_in_modules(model_name, module_names):
    """Check if a model exists within a subset of modules
    Args:
        model_name (str) - name of model to check
        module_names (tuple, list, set) - names of modules to search in
    """
    assert isinstance(module_names, (tuple, list, set))
    return any(model_name in _module_to_models[n] for n in module_names)


def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained


def _timm_pretrained_models():
    """Returns list of models with pretrained weights in timm library."""
    import timm

    models = timm.list_models(pretrained=True)
    return set(models)


def _to_timm_module_name(module):
    """
    Some modules are called differently in tfimm and timm. This function converts the
    tfimm name to the timm name.
    """
    if module == "vit":
        module = "vision_transformer"
    elif module == "swin":
        module = "swin_transformer"
    return module


def _compare_available_models_with_timm(
    name_filter: Union[str, List[str]] = "",
    module: str = "",
    exclude_filters: Union[str, List[str]] = "",
):
    """Helper function to list which models have not yet been converted from timm."""
    import timm

    tf_models = list_models(
        name_filter=name_filter,
        module=module,
        pretrained="timm",
        exclude_filters=exclude_filters,
    )
    pt_models = timm.list_models(
        filter=name_filter,
        module=_to_timm_module_name(module),
        pretrained=True,
        exclude_filters=exclude_filters,
    )

    pt_only = sorted(list(set(pt_models) - set(tf_models)))
    print(f"timm models available in tfimm: {len(tf_models)}/{len(pt_models)}.")
    print(f"timm models not available: {len(pt_only)}.")
    print(f"The following models are not available: {', '.join(pt_only)}")