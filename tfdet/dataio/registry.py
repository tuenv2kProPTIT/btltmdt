from dataclasses import dataclass



_all_pipeline={}


def register(cls):
    cfg = cls.cfg_class
    name = cfg.name 
    name=name.lower()
    if name not in _all_pipeline:
        _all_pipeline[name]={
            'config':cfg,
            'instance':cls 
        }
        