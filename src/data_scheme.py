from datetime import datetime
from typing import List, Optional
from loguru import logger

class Task:
    """
    Represents a schedulable task in the production system.

    Attributes:
      task_id (int): Unique identifier for the task.
      name (str): A short name or description of the task.
      workload (float): Duration the task requires (minutes, hours, etc.).
      mandatory_equip_count (int): How many equipment units of matching category
          must be allocated simultaneously (e.g., 1 isolator).
      optional_equip_count (int): How many additional optional equipment units
          can be allocated if available (0 if none).
      required_workers (int): How many workers are needed to perform this task
          (or how many worker "slots" must be filled).
      required_skill (str): Required worker skill/team name. 
          e.g. "opérateur", "Classificateur".
      start_earliest (datetime): Earliest possible start time (hard constraint).
      end_latest (datetime): Latest permissible end time (hard constraint).
      predecessor_ids (List[int]): List of task_ids that must finish
          before this task starts.
      zone_id (int): Optional zone in which the task occurs (if relevant).
      sequence (int): A numeric sequence index (if tasks have a known order).
      category_equip (str): Equipment category needed (e.g., "LYO", "ISOLATOR").
      priority (int): Optional priority level; a smaller number might
          mean higher priority. (Newly added)
      due_date (datetime): Optional due date to finish this task; 
          could be used for an objective function or constraints. (Newly added)
      comments (str): Optional free-text field for notes. (Newly added)
    """

    def __init__(
        self,
        process_id,
        process_name,
        task_id: int,
        name: str,
        workload: float,
        # Equipment constraints
        mandatory_equip_count: int = 0,
        optional_equip_count: int = 0,
        # Worker constraints
        required_workers: int = 0,
        required_skill: Optional[str] = None,
        # Time constraints
        start_earliest: Optional[datetime] = None,
        end_latest: Optional[datetime] = None,
        # Precedence constraints
        predecessor_ids: Optional[List[int]] = None,
        # Spatial / zone
        zone_id: Optional[int] = None,
        # Sequence
        sequence: Optional[int] = None,
        # Equipment category
        category_equip: Optional[str] = None,
        # Additional fields
        priority: Optional[int] = None,
        due_date: Optional[datetime] = None,
        comments: Optional[str] = None
    ):
        self.process_id = process_id
        self.process_name = process_name
        self.task_id = task_id
        self.name = name
        self.workload = workload

        self.mandatory_equip_count = mandatory_equip_count
        self.optional_equip_count = optional_equip_count

        self.required_workers = required_workers
        self.required_skill = required_skill

        self.start_earliest = start_earliest
        self.end_latest = end_latest
        self.predecessor_ids = predecessor_ids if predecessor_ids else []

        self.zone_id = zone_id
        self.sequence = sequence
        self.category_equip = category_equip

        # Newly added attributes
        self.priority = priority
        self.due_date = due_date
        self.comments = comments

        logger.info(f"Task created: {self}")

    def __repr__(self):
        return (
            f"Task(task_id={self.task_id}, name='{self.name}', "
            f"workload={self.workload}, "
            f"equip_counts=[{self.mandatory_equip_count}-"
            f"{self.mandatory_equip_count + self.optional_equip_count}], "
            f"required_workers={self.required_workers}, skill={self.required_skill}, "
            f"earliest={self.start_earliest}, latest={self.end_latest}, "
            f"predecessors={self.predecessor_ids}, zone_id={self.zone_id}, "
            f"sequence={self.sequence}, category_equip={self.category_equip}, "
            f"priority={self.priority}, due_date={self.due_date}, "
            f"comments='{self.comments}')"
        )

class Resource:
    """
    Represents a resource in the production system, such as a worker or equipment.

    Attributes:
      resource_id (int): Unique identifier for the resource.
      resource_type (str): Type of resource, e.g. "WORKER", "EQUIPMENT", or "ZONE".
      name (str): Name or label for the resource.
      capacity (int): If this resource can handle multiple tasks at once
          (e.g., a machine that runs 2 lines simultaneously). Usually 1.
      shifts (List[tuple]): For WORKER, a list of (start_min, end_min, date)
          specifying availability windows. For equipment, can store maintenance
          windows or downtime similarly if needed.
      skills (List[str]): If this is a worker, store any skills
          (e.g., ["opérateur", "Classificateur"]).
      equipment_category (str): If this is an equipment resource,
          store the equipment's category (e.g., "LYO", "ISOLATOR"). (Newly added)
      location (str): Optional text specifying the resource's location
          or department. (Newly added)
      notes (str): Optional free-text field for extra info. (Newly added)
    """

    def __init__(
        self,
        resource_id: int,
        resource_type: str,
        name: str,
        capacity: int = 1,
        shifts: Optional[List] = None,
        skills: Optional[List[str]] = None,
        # Newly added
        equipment_category: Optional[str] = None,
        location: Optional[str] = None,
        notes: Optional[str] = None
    ):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.name = name
        self.capacity = capacity

        self.shifts = shifts if shifts else []
        self.skills = skills if skills else []

        # For equipment
        self.equipment_category = equipment_category
        # Additional
        self.location = location
        self.notes = notes

        # logger.info(f"Resource created: {self}")

    def __repr__(self):
        return (
            f"Resource(id={self.resource_id}, type={self.resource_type}, "
            f"name='{self.name}', capacity={self.capacity}, "
            f"skills={self.skills}, equipment_category={self.equipment_category}, "
            f"location='{self.location}', notes='{self.notes}')"
        )
