from dataclasses import dataclass
from enum import Enum, auto
import numpy 
DEBUG = True

# Regions are defined to specify search area for template matching, 
# it follows formula where two oposing corners are specified
# so x1,y1,x2,y2 where x1,y1 is top left and x2,y2 is bottom right
region1 = (1027/1280, 563/720, 233/1280, 140/720)
region2 = (250/1280, 590/720, 650/1280, 110/720)

# Paths to images for templates
TEMPLATES = {
    "advanced_building_menu": ("./assets/advanced_building_menu.png", region1),
    "basic_building_menu": ("./assets/basic_building_menu_2.png", region1),
    "build_selection": ("./assets/build_selection.png", region1),
    "test_to_spray_3": ("./assets/test_to_spray_3.png", region1),
    "worker_selected": ("./assets/worker_selected.png", region2)
}

class BuildingState(Enum):
    IDLE = auto()
    WORKER_SELECTED = auto()
    BUILDING_MENU = auto()
    PLACE_BUILDING = auto()
    FINISHED = auto()

@dataclass
class FrameResult:
    frame: int

@dataclass
class Template:
    name:str
    region:tuple[int,int]
    template_grey:numpy.ndarray

