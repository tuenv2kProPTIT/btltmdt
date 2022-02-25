from tfdet.core.config import CoreConfig
from dataclasses import dataclass


@dataclass
class PseudoSamplerConfig(CoreConfig):
    name='PseudoSampler'


class PseudoSampler:
    cfg_class=PseudoSamplerConfig
    def __init__(self, cfg:PseudoSamplerConfig,*args, **kwargs) -> None:
        self.cfg = cfg 
    
    def sampler(self, matched_index):
        return matched_index