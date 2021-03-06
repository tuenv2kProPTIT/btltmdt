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
    return cls
def get_pipeline(cfg):
    name = cfg['name'].lower()
    if name in _all_pipeline:
        dict_cls = _all_pipeline[name]
        try:
            return dict_cls['instance'](dict_cls['config'](**cfg))
        except Exception as e:
            print(e)
            raise ValueError(cfg)
    else:
        print(f"pipeline with {name} didn't register anywhere")
        raise ValueError(cfg)