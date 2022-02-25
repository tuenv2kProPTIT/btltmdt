from dataclasses import dataclass
@dataclass
class BackBoneConfig:
    """Share core config 
    """

    name: str = ""
    url : str = ""
    last_modified: str = ""

@dataclass
class NeckConfig:
    """Share core config 
    """

    name: str = ""
    url : str = ""
    last_modified: str = ""

@dataclass
class HeadConfig:
    """Share HeadDensenConfig
    """
    name: str=""
    url : str=""
    last_modified: str = ""