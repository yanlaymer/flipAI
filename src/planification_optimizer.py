from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger
from src.data_scheme import Task, Resource
from typing import List, Optional

def build_and_solve_plan_model(
    tasks: List[Task],
    resources: List[Resource],
    time_unit: str = "minute",
    buffer_ratio: float = 0.2,
    max_solver_time_s: float = 60.0,
    preferred_start_time: Optional[datetime] = None  # Add this parameter
):
    logger.info("Starting to build and solve the schedule model.")

    # -----------------------------------------------------------
    # (A) Optionally adjust tasks so that if a task has a defined
    #     equipment category or required skill but a count of 0,
    #     we bump it up to 1.
    # -----------------------------------------------------------
    # for t in tasks:
        # If the task specifies an equipment category but no count then require 1 unit.
        # if t.category_equip and (t.mandatory_equip_count + t.optional_equip_count) == 0:
        #     logger.debug(f"Adjusting task {t.task_id} equipment count from 0 to 1")
        #     t.mandatory_equip_count = 

        # Similarly, if you expect tasks to have at least one worker when a skill is needed
        # (or simply by default), update t.required_workers if it is 0.
        # (Adjust this logic to your domain – if a task can be performed without a worker then leave it.)
        # if t.required_workers == 0:
        #     logger.debug(f"Adjusting task {t.task_id} worker count from 0 to 1")
        #     t.required_workers = 1

        # Optionally, if your tasks should have a required skill but the field is empty,
        # you could set a default value here. For example:
        # if not hasattr(t, 'required_skill') or t.required_skill is None:
        #     t.required_skill = ""  # Meaning "no specific skill required"
    
    # Determine global earliest based on preferred_start_time or tasks
    if preferred_start_time is not None:
        global_earliest = preferred_start_time
    else:
        valid_starts = [t.start_earliest for t in tasks if t.start_earliest is not None]
        global_earliest = min(valid_starts) if valid_starts else datetime.now()

    logger.info("Global earliest start set to: {}", global_earliest)
    
    def dt_to_int(dt: Optional[datetime]) -> int:
        if dt is None:
            return 0
        delta = dt - global_earliest
        minutes = int(delta.total_seconds() // 60)
        return minutes if time_unit == "minute" else minutes // 60

    # Compute the horizon (sum of workloads plus a buffer)
    sum_workloads = sum(int(t.workload) for t in tasks)
    buffer_amount = int(buffer_ratio * sum_workloads)
    HORIZON = sum_workloads + buffer_amount
    if HORIZON < 1:
        HORIZON = 100_000  # must not be zero
    logger.info("Computed horizon: {}", HORIZON)

    # Separate resources by type
    eq_list = [r for r in resources if r.resource_type == "EQUIPMENT"]
    wr_list = [r for r in resources if r.resource_type == "WORKER"]
    logger.info("Separated resources by type: {} equipment, {} workers", len(eq_list), len(wr_list))

    # Create the CP model
    model = cp_model.CpModel()
    logger.info("Building the CP model.")

    # 1) Create interval variables for tasks
    task_vars = {}  # task_id -> (start_var, duration, end_var, interval_var)
    for t in tasks:
        duration = int(t.workload)
        start_var = model.NewIntVar(0, HORIZON, f"start_{t.task_id}")
        end_var = model.NewIntVar(0, HORIZON, f"end_{t.task_id}")
        interval_var = model.NewIntervalVar(start_var, duration, end_var, f"interval_{t.task_id}")
        task_vars[t.task_id] = (start_var, duration, end_var, interval_var)
    logger.info("Created interval variables for tasks.")

    # 2) Equipment usage constraints
    equip_used = {}
    for t in tasks:
        for eq in eq_list:
            if t.category_equip and eq.equipment_category == t.category_equip:
                var = model.NewBoolVar(f"equip_used_{t.task_id}_{eq.resource_id}")
            else:
                var = model.NewBoolVar(f"equip_used_{t.task_id}_{eq.resource_id}")
                model.Add(var == 0)
            equip_used[(t.task_id, eq.resource_id)] = var

    # Add equipment count constraints (only one set now)
    for t in tasks:
        if t.mandatory_equip_count + t.optional_equip_count > 0:
            relevant_eq = [eq.resource_id for eq in eq_list if eq.equipment_category == t.category_equip]
            model.Add(
                sum(equip_used[(t.task_id, eid)] for eid in relevant_eq)
                >= t.mandatory_equip_count
            )
            model.Add(
                sum(equip_used[(t.task_id, eid)] for eid in relevant_eq)
                <= t.mandatory_equip_count + t.optional_equip_count
            )

    # NoOverlap constraints for equipment resources
    eq_intervals_per_e = {eq.resource_id: [] for eq in eq_list}
    for t in tasks:
        s_var, dur, e_var, _ = task_vars[t.task_id]
        for eq in eq_list:
            bool_var = equip_used[(t.task_id, eq.resource_id)]
            opt_interval = model.NewOptionalIntervalVar(s_var, dur, e_var, bool_var,
                                                        f"eq_opt_int_{t.task_id}_{eq.resource_id}")
            eq_intervals_per_e[eq.resource_id].append(opt_interval)
    for eq in eq_list:
        model.AddNoOverlap(eq_intervals_per_e[eq.resource_id])
    logger.info("Added equipment usage constraints.")

    # 3) Worker usage constraints
    worker_used = {}
    for t in tasks:
        for w in wr_list:
            # If a task has a required skill and the worker does not have it, force assignment = 0.
            # (If t.required_skill is empty (or None), then any worker is allowed.)
            if hasattr(t, 'required_skill') and t.required_skill and t.required_skill not in w.skills:
                var = model.NewBoolVar(f"worker_used_{t.task_id}_{w.resource_id}")
                model.Add(var == 0)
            else:
                var = model.NewBoolVar(f"worker_used_{t.task_id}_{w.resource_id}")
            worker_used[(t.task_id, w.resource_id)] = var

    # Add worker count constraints (only add one set of constraints)
    for t in tasks:
        if t.required_workers > 0:
            model.Add(
                sum(worker_used[(t.task_id, w.resource_id)] for w in wr_list)
                == t.required_workers
            )

    # NoOverlap constraints for worker resources
    w_intervals_per_w = {w.resource_id: [] for w in wr_list}
    for t in tasks:
        s_var, dur, e_var, _ = task_vars[t.task_id]
        for w in wr_list:
            bool_var = worker_used[(t.task_id, w.resource_id)]
            opt_interval = model.NewOptionalIntervalVar(s_var, dur, e_var, bool_var,
                                                        f"w_opt_int_{t.task_id}_{w.resource_id}")
            w_intervals_per_w[w.resource_id].append(opt_interval)
    for w in wr_list:
        model.AddNoOverlap(w_intervals_per_w[w.resource_id])
    logger.info("Added worker usage constraints.")

    # 4) Precedence constraints
    for t in tasks:
        s_var, _, e_var, _ = task_vars[t.task_id]
        for p_id in t.predecessor_ids:
            if p_id in task_vars:
                pred_end = task_vars[p_id][2]
                model.Add(s_var >= pred_end)
    logger.info("Added precedence constraints.")

    # 5) Sequence constraints for tasks sharing the same zone (if any)
    zone_dict = defaultdict(list)
    for t in tasks:
        if t.zone_id is not None and t.sequence is not None:
            zone_dict[t.zone_id].append(t)
    for z_id, z_tasks in zone_dict.items():
        z_tasks_sorted = sorted(z_tasks, key=lambda x: x.sequence)
        for i in range(len(z_tasks_sorted) - 1):
            cur_t = z_tasks_sorted[i]
            nxt_t = z_tasks_sorted[i+1]
            model.Add(task_vars[nxt_t.task_id][0] >= task_vars[cur_t.task_id][2])
    logger.info("Added sequence constraints.")

    # 6) Minimize makespan
    all_end_vars = [task_vars[t.task_id][2] for t in tasks]
    makespan = model.NewIntVar(0, HORIZON, "makespan")
    model.AddMaxEquality(makespan, all_end_vars)
    model.Minimize(makespan)
    logger.info("Minimizing makespan.")

    # 7) Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_solver_time_s
    logger.info("Solving the model.")
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.info("No feasible solution found.")
        return {}, None
    logger.info("Solver status: {}", solver.StatusName(status))

    # 8) Build the final schedule output
    schedule = {"by_task": {}, "by_resource": {}}
    for r in resources:
        schedule["by_resource"][r.resource_id] = []

    def int_to_dt(offset: int) -> datetime:
        return (global_earliest + timedelta(minutes=offset)) if time_unit == "minute" \
               else (global_earliest + timedelta(hours=offset))

    for t in tasks:
        s_var, dur, e_var, _ = task_vars[t.task_id]
        start_val = solver.Value(s_var)
        end_val = solver.Value(e_var)
        real_start = int_to_dt(start_val)
        real_end = int_to_dt(end_val)

        used_equip = [eq.resource_id for eq in eq_list
                      if solver.Value(equip_used[(t.task_id, eq.resource_id)]) == 1]
        used_workers = [w.resource_id for w in wr_list
                        if solver.Value(worker_used[(t.task_id, w.resource_id)]) == 1]

        schedule["by_task"][t.task_id] = {
            "process_id": t.process_id,
            "process_name": t.process_name,
            "task_id": t.task_id,
            "task_name": t.name,
            "start_int": start_val,
            "end_int": end_val,
            "start_optimized": real_start,
            "end_optimized": real_end,
            "used_equipment": used_equip,
            "used_workers": used_workers,
            "workload": t.workload,
            "team": getattr(t, 'required_skill', None),
            "sequence": t.sequence,
            "zone_id": t.zone_id,
            "initial_start": t.start_earliest,
            "initial_end": t.end_latest
        }

    # Assign task intervals to resources
    for t in tasks:
        s_var, dur, e_var, _ = task_vars[t.task_id]
        start_val = solver.Value(s_var)
        end_val = solver.Value(e_var)
        real_start = int_to_dt(start_val)
        real_end = int_to_dt(end_val)
        for eq in eq_list:
            if solver.Value(equip_used[(t.task_id, eq.resource_id)]) == 1:
                schedule["by_resource"][eq.resource_id].append({
                    "task_id": t.task_id,
                    "task_name": t.name,
                    "start_int": start_val,
                    "end_int": end_val,
                    "start_optimized": real_start,
                    "end_optimized": real_end,
                    "initial_start": t.start_earliest,
                    "initial_end": t.end_latest,
                    "mandatory_count": t.mandatory_equip_count,
                    "optional_count": t.optional_equip_count,
                })
        for w in wr_list:
            if solver.Value(worker_used[(t.task_id, w.resource_id)]) == 1:
                schedule["by_resource"][w.resource_id].append({
                    "task_id": t.task_id,
                    "task_name": t.name,
                    "start_int": start_val,
                    "end_int": end_val,
                    "start_optimized": real_start,
                    "end_optimized": real_end,
                    "team": getattr(t, 'required_skill', None)
                })

    # Sort each resource’s intervals by start time
    for r_id, intervals in schedule["by_resource"].items():
        intervals.sort(key=lambda x: x["start_int"])

    makespan_val = solver.Value(makespan)
    logger.info("Schedule built successfully with makespan: {}", makespan_val)
    print("Makespan (int) =", makespan_val)
    print("Makespan start_dt => end_dt =", int_to_dt(0), "=>", int_to_dt(makespan_val))

    return schedule, makespan_val
