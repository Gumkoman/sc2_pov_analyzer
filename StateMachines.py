from enum import Enum,auto


class State(Enum):
    IDLE = auto()
    WORKER_SELECTED = auto()
    BUILDING_SELECTION = auto()
    PLACE_SELECTION = auto()
    FINISHED = auto()







def execute_state_machine(matching_list,thresholds):
    result = []
    procedure_timer = 0
    current_state = State.IDLE
    for i,element in enumerate(matching_list):
        procedure_timer += 1
        if current_state == State.IDLE:
            procedure_timer = 0
            if element["worker_selected"] > thresholds["worker_selected"]:
                current_state = State.WORKER_SELECTED
        elif current_state == State.WORKER_SELECTED:
            if (element["basic_building_menu"] > thresholds["basic_building_menu"] 
                    or element["advanced_building_menu"] > thresholds["advanced_building_menu"]):
                    current_state = State.BUILDING_SELECTION
            elif element["worker_selected"] < thresholds["worker_selected"]:
                current_state = State.IDLE
        elif current_state == State.BUILDING_SELECTION:
            if  (element["basic_building_menu"] < thresholds["basic_building_menu"] 
                and element["advanced_building_menu"] < thresholds["advanced_building_menu"]):
                if element["build_selection"] > thresholds["build_selection"]:
                    current_state = State.PLACE_SELECTION
                else:
                    current_state = State.IDLE
        elif current_state == State.PLACE_SELECTION:
            if element["build_selection"] < thresholds["build_selection"]:
                 current_state = State.FINISHED
        elif current_state == State.FINISHED:
            result.append({
                "start": i-procedure_timer,
                "end": i,
                "procedure_time" : procedure_timer
            }) 
            current_state = State.IDLE
    return result