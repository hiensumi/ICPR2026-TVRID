# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .AirportALERT import AirportALERT
from .iLIDS import iLIDS
from .pku import PKU
from .prai import PRAI
from .prid import PRID
from .grid import GRID
from .saivt import SAIVT
from .sensereid import SenseReID
from .sysu_mm import SYSU_mm
from .thermalworld import Thermalworld
from .pes3d import PeS3D
from .caviara import CAVIARa
from .viper import VIPeR
from .lpw import LPW
from .shinpuhkan import Shinpuhkan
from .wildtracker import WildTrackCrop
from .cuhk_sysu import cuhkSYSU

# Vehicle re-id datasets
from .veri import VeRi
from .vehicleid import VehicleID, SmallVehicleID, MediumVehicleID, LargeVehicleID
from .veriwild import VeRiWild, SmallVeRiWild, MediumVeRiWild, LargeVeRiWild

# TVRID dataset for ICPR 2026 competition
from .tvrid import (
    TVRID_RGB, TVRID_Depth, TVRID_Cross,
    TVRID_Depth_CombinedSplit, TVRID_Depth_CombinedSplit_DBStratified,
    TVRID_Depth_CombinedSplit_DBHalf,
    TVRID_RGB_CombinedSplit, TVRID_RGB_CombinedSplit_DBStratified,
    TVRID_RGB_DBOnlySplit, TVRID_RGB_CombinedSplit_DBHalf,
    TVRID_RGB_DBOnlySplit_MultiFrame,
    TVRID_RGB_DBPublic_MultiFrame,
    TVRID_RGB_TVPR2OnlySplit_MultiFrame,
    TVRID_RGB_DB_AllVal, TVRID_RGB_CrossCameraVal,
)

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
