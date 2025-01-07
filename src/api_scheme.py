from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class OptimizeRequest(BaseModel):
    """Represents the query parameters or a possible request body.
       Here we only show 'start' and 'end' as query params, so you might
       not strictly need a Request model. But included for completeness."""
    start: datetime = Field(..., description="Start of scheduling window (minutes precision)")
    end:   datetime = Field(..., description="End of scheduling window (minutes precision)")

class ScheduledTaskOut(BaseModel):
    """Represents the scheduled details for a single task."""
    process_id: int
    process_name: Optional[str]
    task_id: int
    task_name: str
    start_optimized: datetime
    end_optimized: datetime
    used_equipment: List[int]
    used_workers: List[int]
    workload: float
    team: Optional[str]
    sequence: Optional[int]
    zone_id: Optional[int]
    initial_start: Optional[datetime]
    initial_end: Optional[datetime]

class ScheduledResourceIntervalOut(BaseModel):
    """One scheduled interval on a resource timeline."""
    task_id: int
    task_name: str
    start_optimized: datetime
    end_optimized: datetime
    # Possibly more fields:
    team: Optional[str] = None
    initial_start: Optional[datetime] = None
    initial_end: Optional[datetime] = None
    mandatory_count: Optional[int] = None
    optional_count: Optional[int] = None

class ScheduleResult(BaseModel):
    """Overall scheduling result: tasks + resources usage."""
    by_task: Dict[str, ScheduledTaskOut]
    by_resource: Dict[str, List[ScheduledResourceIntervalOut]]
    makespan: Optional[int] = None