
import imp
from tfdet.dataio.registry import get_pipeline
from tfdet.dataio.transform import Compose


def pipeline(cfgs):
    assert len(cfgs) > 0 
    cfg_0 = cfgs[0]
    assert cfg_0['name'].lower() == 'InputReadRecordFiles'.lower()
    data  = get_pipeline(cfg_0)() # load instance
    cfgs = cfgs[1:] 
    pipes = [
        get_pipeline(cfg) for cfg in cfgs
    ]
    for pipe in pipes:
        config = pipe.get_config()
        num_parallel_calls=config['num_parallel_calls']
        deterministic=config['deterministic']
        data = data.map(pipe, num_parallel_calls=num_parallel_calls, deterministic = deterministic)
    
    return data 
    