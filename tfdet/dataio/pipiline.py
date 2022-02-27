
import imp
from tfdet.dataio.registry import get_pipeline



def pipeline(cfg):  
    return get_pipeline(cfg)