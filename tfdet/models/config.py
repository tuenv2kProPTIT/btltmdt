from dataclasses import dataclass
@dataclass
class BackBoneConfig:
    """Share core config 
    """

    name: str = ""
    url : str = ""
    last_modified: str = ""

