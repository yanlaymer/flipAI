from fastapi import FastAPI, Query, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

# Import your modules
from src.gather_tasks import gather_all_scheduling_data
from src.data_scheme import Task, Resource   # your custom data classes
from src.optimizer_with_workers import build_and_solve_schedule_model
from src.build_task_resources import build_tasks_and_resources
from src.api_scheme import ScheduledTaskOut, ScheduledResourceIntervalOut, ScheduleResult

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create database engine and scheduler
    app.state.db_engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=10
    )
    app.state.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=app.state.db_engine)
    yield
    # Cleanup: Close database connections
    app.state.db_engine.dispose()

app = FastAPI(
    title="FlipSoft Optimizer AI API",
    description="Optimizes initial tasks with equipment, workers, and zones using CP-SAT.",
    version="1.0.0",
    lifespan=lifespan
)

# Database configuration
DATABASE_URL = "mssql+pyodbc://oleksii:sdjnn4393vn@194.76.26.191/FLIPCLOUD?driver=ODBC+Driver+17+for+SQL+Server"

# Dependency to get the database engine
def get_db():
    engine = app.state.db_engine
    with engine.connect() as conn:
        yield conn


@app.get("/api/schedule", response_model=ScheduleResult)
def optimize_schedule(
    start: datetime = Query(..., description="Start of scheduling window, e.g. 2024-07-15T06:00"),
    end:   datetime = Query(..., description="End of scheduling window, e.g. 2024-07-16T06:00"),
    engine = Depends(get_db)
):
    """
    Main scheduling endpoint:
    1) Gathers raw data from DB (gather_all_scheduling_data).
    2) Converts raw data -> Task/Resource objects.
    3) Optimizes via build_and_solve_schedule_model.
    4) Returns the final schedule in structured JSON.
    """

    # Hardcoded scope_id for demo; set as needed:
    scope_id = 26

    # 1) Gather raw data
    raw_data = gather_all_scheduling_data(engine, scope_id, start, end, max_tasks=50)
    if not raw_data.get("processes"):
        # No processes found -> nothing to schedule
        return ScheduleResult(by_task={}, by_resource={}, makespan=None)

    # 2) Convert raw data -> domain objects
    tasks, resources = build_tasks_and_resources(raw_data)

    # 3) Solve
    schedule_dict, makespan_val = build_and_solve_schedule_model(
        tasks=tasks,
        resources=resources,
        time_unit="minute",
        buffer_ratio=0.2,
        max_solver_time_s=60.0
    )
    if not schedule_dict:
        # Infeasible solution
        raise HTTPException(status_code=400, detail="No feasible solution found.")

    # 4) Build the Pydantic-friendly response
    #    schedule_dict has shape: {"by_task": {...}, "by_resource": {...}}
    #    We must transform them into ScheduleResult structure.

    # (A) by_task
    # schedule_dict["by_task"] is typically: { task_id: {...fields...}, ... }
    pyd_tasks = {}
    for t_id_str, t_info in schedule_dict["by_task"].items():
        # t_id_str is often an int or string
        # We create a ScheduledTaskOut object
        pyd_obj = ScheduledTaskOut(
            process_id=t_info["process_id"],
            process_name=t_info["process_name"],
            task_id=t_info["task_id"],
            task_name=t_info["task_name"],
            start_optimized=t_info["start_optimized"],
            end_optimized=t_info["end_optimized"],
            used_equipment=t_info["used_equipment"],
            used_workers=t_info["used_workers"],
            workload=t_info["workload"],
            team=t_info["team"],
            sequence=t_info["sequence"],
            zone_id=t_info["zone_id"],
            initial_start=t_info["initial_start"],
            initial_end=t_info["initial_end"]
        )
        # We'll store it with a string key or int key.
        pyd_tasks[str(t_info["task_id"])] = pyd_obj

    # (B) by_resource
    # schedule_dict["by_resource"] is typically { resource_id: [ {task_id..., start_optimized..., etc.}, ... ], ... }
    pyd_resources = {}
    for r_id, intervals in schedule_dict["by_resource"].items():
        pyd_list = []
        for inter in intervals:
            pyd_inter = ScheduledResourceIntervalOut(
                task_id=inter["task_id"],
                task_name=inter["task_name"],
                start_optimized=inter["start_optimized"],
                end_optimized=inter["end_optimized"],
                team=inter.get("team"),
                initial_start=inter.get("initial_start"),
                initial_end=inter.get("initial_end"),
                mandatory_count=inter.get("mandatory_count"),
                optional_count=inter.get("optional_count")
            )
            pyd_list.append(pyd_inter)
        pyd_resources[str(r_id)] = pyd_list

    # Final
    out = ScheduleResult(
        by_task=pyd_tasks,
        by_resource=pyd_resources,
        makespan=makespan_val
    )
    return out
