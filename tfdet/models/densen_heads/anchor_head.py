from dataclasses import dataclass
from re import T
from tfdet.models.config import HeadConfig
from tfdet.core.anchor.anchor_generator import AnchorConfig, AnchorGenerator
from typing import Dict

@dataclass
class AnchorHeadConfig(HeadConfig):
    anchor_config: Dict = None 
    assigner: Dict  = None 
    sampler : Dict = None
    

    
    