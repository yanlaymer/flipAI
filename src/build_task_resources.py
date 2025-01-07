import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.data_scheme import Task, Resource
from loguru import logger

# Assume you have your enhanced data classes in the same file or imported:
# from my_data_classes import Task, Resource

def build_tasks_and_resources(data: Dict[str, Any]):
    logger.info("Building tasks and resources from raw data.")
    """
    Converts raw dictionary data into lists of Task and Resource objects,
    with no reference-date logic. We keep datetime fields as-is.

    data is expected to have:
      - data["processes"] (list of process/activity rows)
      - data["equipment"] (list of equipment rows)
      - data["zones"] (list of zone rows)
      - data["workers"] (dict {worker_id: {...}})
      - data["shifts"] (dict {worker_id: [(start_min, end_min, shift_date), ...]})

    Returns:
      tasks: List[Task]
      resources: List[Resource]
    """

    # -------------------------------------------------------------------------
    # 1) Parse Tasks from data["processes"]
    #    Potentially, multiple rows can share the same act_id with different
    #    resource references. We'll accumulate them in a dict keyed by act_id.
    # -------------------------------------------------------------------------
    logger.info("Parsing tasks from processes data.")
    tasks_dict = {}

    for row in data["processes"]:
        act_id = row["act_id"]

        if act_id not in tasks_dict:
            # Initialize a skeletal dictionary for this task
            tasks_dict[act_id] = {
                "process_id": row["process_id"],
                "process_name": row["process_name"],
                "task_id": act_id,
                "name": str(row["act_name"]),
                "workload": row.get("act_workload", 0) or 0,
                "mandatory_equip_count": 0,
                "optional_equip_count": 0,
                "required_workers": 0,
                "required_skill": None,
                "start_earliest": row.get("start_date", None),   # keep as datetime
                "end_latest": row.get("end_date", None),         # keep as datetime
                "predecessor_ids": [],
                "zone_id": None,
                "sequence": row.get("pra_seq", None),
                "category_equip": row.get("equipment_category", None),
            }

            # Parse any "requiredWorkers" JSON to detect how many
            # workers or what skill might be needed
            try:
                workers_json = json.loads(row.get("requiredWorkers", "[]"))
                total_req_workers = 0
                skill_detected = None
                for wj in workers_json:
                    # e.g. if wj["RE_TYPE"] == "TEAM":
                    total_req_workers += wj.get("ACR_QTY", 0)
                    # skill might come from wj["RE_NAME"], e.g. "opérateur"
                    if not skill_detected and "RE_NAME" in wj:
                        skill_detected = wj["RE_NAME"]
                tasks_dict[act_id]["required_workers"] = total_req_workers
                tasks_dict[act_id]["required_skill"]   = skill_detected

            except json.JSONDecodeError:
                pass

        # Now parse resource references from the row:
        resource_type = row["resource_type"]
        resource_option = row.get("resource_option", "auto")

        if resource_type == "EQUIPEMENT":
            # If resource_option == "required", increment mandatory
            if resource_option == "required":
                tasks_dict[act_id]["mandatory_equip_count"] += 1
            else:
                tasks_dict[act_id]["optional_equip_count"]  += 1

        elif resource_type == "ZONE":
            # We can store the zone_id if not already set
            if tasks_dict[act_id]["zone_id"] is None:
                tasks_dict[act_id]["zone_id"] = row["resource_id"]

        elif resource_type == "TEAM":
            # Possibly we already accounted for it in requiredWorkers JSON
            pass

        # If you have any "predecessor" logic in these rows,
        # you could append them here:
        # predecessor_val = row.get("some_predecessor_column")
        # if predecessor_val:
        #     tasks_dict[act_id]["predecessor_ids"].append(predecessor_val)

    # Convert tasks_dict → list of Task objects
    tasks = []
    for act_id, td in tasks_dict.items():
        t = Task(
            process_id           = td["process_id"],
            process_name         = td["process_name"],
            task_id              = td["task_id"],
            name                 = td["name"],
            workload             = td["workload"],
            mandatory_equip_count= td["mandatory_equip_count"],
            optional_equip_count = td["optional_equip_count"],
            required_workers     = td["required_workers"],
            required_skill       = td["required_skill"],
            start_earliest       = td["start_earliest"],  # kept as raw datetime
            end_latest           = td["end_latest"],      # kept as raw datetime
            predecessor_ids      = td["predecessor_ids"],
            zone_id              = td["zone_id"],
            sequence             = td["sequence"],
            category_equip       = td["category_equip"]
        )
        tasks.append(t)

    logger.info("Parsed {} tasks.", len(tasks))
    # -------------------------------------------------------------------------
    # 2) Build Resource objects from equipment, zones, workers
    # -------------------------------------------------------------------------
    logger.info("Building resources from equipment, zones, and workers data.")
    resources: List[Resource] = []

    # (A) Equipment
    for eq_row in data.get("equipment", []):
        r = Resource(
            resource_id       = eq_row["equipment_id"],
            resource_type     = "EQUIPMENT",
            name              = eq_row["equipment_name"],
            capacity          = 1,  # or parse if data has a capacity
            shifts            = None,  # often no shift data for eq
            skills            = None,
            equipment_category= eq_row.get("category", None),
            location          = None,
            notes             = None
        )
        resources.append(r)

    # (B) Zones
    for z_row in data.get("zones", []):
        r = Resource(
            resource_id       = z_row["zone_id"],
            resource_type     = "ZONE",
            name              = z_row["zone_name"],
            capacity          = 1,
            shifts            = None,
            skills            = None,
            equipment_category= z_row.get("category", None),
            location          = None,
            notes             = None
        )
        resources.append(r)

    # (C) Workers
    # data["workers"] is a dict: { worker_id -> {...} }
    # data["shifts"] is: { worker_id -> [(start_min, end_min, shift_date), ...] }
    for w_id, w_info in data.get("workers", {}).items():
        # We'll interpret w_info["team_name"] as their skill, etc.
        skill_list = []
        if w_info.get("team_name"):
            skill_list = [ w_info["team_name"] ]

        # Grab shift data as is—no day0 offset
        shift_data = data.get("shifts", {}).get(w_id, [])

        r = Resource(
            resource_id       = w_id,
            resource_type     = "WORKER",
            name              = f"Worker {w_id}",
            capacity          = 1,
            shifts            = shift_data,
            skills            = skill_list,
            equipment_category= None,
            location          = None,
            notes             = None
        )
        resources.append(r)

    logger.info("Parsed {} resources.", len(resources))
    logger.info("Task and resource building complete.")
    # Done
    return tasks, resources
