from ortools.sat.python import cp_model
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from src.data_scheme import Task, Resource
from loguru import logger

def build_and_solve_schedule_model(
    tasks: List["Task"],         # Your pre-loaded tasks
    resources: List["Resource"], # Your pre-loaded resources
    time_unit: str = "minute",   # or "hour" if you prefer
    buffer_ratio: float = 0.2,   # 20% buffer on sum of workloads
    max_solver_time_s: float = 60.0
) -> (Dict[str, Any], Optional[int]):
    """
    Builds and solves a production-style CP-SAT scheduling model:

    Steps:
      1. Identify the global earliest datetime among all tasks (ignoring those w/o start_earliest).
      2. Convert all relevant datetimes to integer offsets from that global earliest.
      3. Compute horizon = sum_of_workloads + buffer.
      4. Define mandatory/optional equipment usage, equipment category matching,
         worker skill matching, earliest/latest constraints, precedence, no-overlap.
      5. Minimize makespan.
      6. Build final schedule:
         - schedule["by_task"]: info per task (start/end mapped back to real datetimes, etc.)
         - schedule["by_resource"]: intervals for each resource, sorted by start time.

    Returns:
      (schedule_dict, makespan_val)
        schedule_dict has keys "by_task" and "by_resource".
        makespan_val is the final schedule length in integer time units
                     (or None if infeasible).

    Note: If you want sub-minute or sub-hour resolution, you can multiply durations
          and do your own scaling logic. This code uses integer-based CP variables.
    """
    logger.info("Starting to build and solve the schedule model.")
    # ------------------------------------------------------------------
    # A) Identify the global earliest start among tasks
    #    We'll only consider tasks that have a non-None 'start_earliest'.
    # ------------------------------------------------------------------
    valid_starts = [t.start_earliest for t in tasks if t.start_earliest is not None]
    if valid_starts:
        global_earliest = min(valid_starts)
    else:
        # If no tasks have a start_earliest, just pick a reference
        global_earliest = datetime.now()
    logger.info("Identified global earliest start: {}", global_earliest)
    # Helper to convert a datetime dt -> integer offset in minutes/hours from global_earliest
    def dt_to_int(dt: Optional[datetime]) -> int:
        if dt is None:
            # We'll treat no constraint as 0 or horizon-later, handle logic below
            return 0
        delta = dt - global_earliest
        minutes = int(delta.total_seconds() // 60)
        if time_unit == "minute":
            return minutes
        elif time_unit == "hour":
            return minutes // 60
        else:
            raise ValueError("Unsupported time_unit. Use 'minute' or 'hour'.")

    # ------------------------------------------------------------------
    # B) Compute sum_of_workloads + buffer as horizon
    #    We'll interpret 'task.workload' as integer minutes/hours
    # ------------------------------------------------------------------
    sum_workloads = 0
    for t in tasks:
        sum_workloads += int(t.workload)

    buffer_amount = int(buffer_ratio * sum_workloads)
    HORIZON = sum_workloads + buffer_amount
    if HORIZON < 1:
        HORIZON = 100_000  # must not be zero
    logger.info("Computed horizon: {}", HORIZON)
    # ------------------------------------------------------------------
    # Separate resources by type
    # ------------------------------------------------------------------
    eq_list = [r for r in resources if r.resource_type == "EQUIPMENT"]
    wr_list = [r for r in resources if r.resource_type == "WORKER"]
    logger.info("Separated resources by type: {} equipment, {} workers", len(eq_list), len(wr_list))
    # ------------------------------------------------------------------
    # C) Build the CP model
    # ------------------------------------------------------------------
    model = cp_model.CpModel()
    logger.info("Building the CP model.")
    # 1) Create interval variables for tasks
    task_vars = {}  # task_id -> (start_var, duration, end_var, interval_var)
    for t in tasks:
        duration = int(t.workload)
        start_var = model.NewIntVar(0, HORIZON, f"start_{t.task_id}")
        end_var   = model.NewIntVar(0, HORIZON, f"end_{t.task_id}")
        interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                            f"interval_{t.task_id}")
        task_vars[t.task_id] = (start_var, duration, end_var, interval_var)

        # # If we want to enforce earliest start:
        # if t.start_earliest is not None:
        #     earliest_offset = dt_to_int(t.start_earliest)
        #     model.Add(start_var >= earliest_offset)

        # # If we want to enforce latest end:
        # if t.end_latest is not None:
        #     latest_offset = dt_to_int(t.end_latest)
        #     model.Add(end_var <= latest_offset)
    logger.info("Created interval variables for tasks.")
    # 2) Equipment usage constraints
    equip_used = {}
    for t in tasks:
        for eq in eq_list:
            # Check equipment category match
            if t.category_equip and eq.equipment_category != t.category_equip:
                var = model.NewBoolVar(f"equip_used_{t.task_id}_{eq.resource_id}")
                model.Add(var == 0)
                equip_used[(t.task_id, eq.resource_id)] = var
            else:
                var = model.NewBoolVar(f"equip_used_{t.task_id}_{eq.resource_id}")
                equip_used[(t.task_id, eq.resource_id)] = var

    # Sum of equip booleans in [mand, mand + opt]
    for t in tasks:
        eq_ids = [eq.resource_id for eq in eq_list]
        model.Add(sum(equip_used[(t.task_id, e_id)] for e_id in eq_ids)
                  >= t.mandatory_equip_count)
        model.Add(sum(equip_used[(t.task_id, e_id)] for e_id in eq_ids)
                  <= t.mandatory_equip_count + t.optional_equip_count)

    # NoOverlap for each equipment
    eq_intervals_per_e = {eq.resource_id: [] for eq in eq_list}
    for t in tasks:
        (s_var, dur, e_var, interval_var) = task_vars[t.task_id]
        for eq in eq_list:
            bool_var = equip_used[(t.task_id, eq.resource_id)]
            opt_interval = model.NewOptionalIntervalVar(
                s_var, dur, e_var, bool_var,
                f"eq_opt_int_{t.task_id}_{eq.resource_id}"
            )
            eq_intervals_per_e[eq.resource_id].append(opt_interval)

    for eq in eq_list:
        model.AddNoOverlap(eq_intervals_per_e[eq.resource_id])
    logger.info("Added equipment usage constraints.")
    # 3) Worker usage constraints (skill matching + exact number of workers)
    worker_used = {}
    for t in tasks:
        for w in wr_list:
            if t.required_skill and (t.required_skill not in w.skills):
                var = model.NewBoolVar(f"worker_used_{t.task_id}_{w.resource_id}")
                model.Add(var == 0)
                worker_used[(t.task_id, w.resource_id)] = var
            else:
                var = model.NewBoolVar(f"worker_used_{t.task_id}_{w.resource_id}")
                worker_used[(t.task_id, w.resource_id)] = var

    for t in tasks:
        w_ids = [wrk.resource_id for wrk in wr_list]
        model.Add(
            sum(worker_used[(t.task_id, w_id)] for w_id in w_ids)
            == t.required_workers
        )

    # NoOverlap for workers
    w_intervals_per_w = {w.resource_id: [] for w in wr_list}
    for t in tasks:
        (s_var, dur, e_var, interval_var) = task_vars[t.task_id]
        for w in wr_list:
            bool_var = worker_used[(t.task_id, w.resource_id)]
            opt_interval = model.NewOptionalIntervalVar(
                s_var, dur, e_var, bool_var,
                f"w_opt_int_{t.task_id}_{w.resource_id}"
            )
            w_intervals_per_w[w.resource_id].append(opt_interval)

    for w in wr_list:
        model.AddNoOverlap(w_intervals_per_w[w.resource_id])
    logger.info("Added worker usage constraints.")
    # 4) Precedence constraints
    for t in tasks:
        start_var, dur, end_var, interval_var = task_vars[t.task_id]
        for p_id in t.predecessor_ids:
            if p_id in task_vars:
                pred_end = task_vars[p_id][2]
                model.Add(start_var >= pred_end)
    logger.info("Added precedence constraints.")
    # 5) Sequence constraints if tasks share the same zone + have a 'sequence'
    from collections import defaultdict
    zone_dict = defaultdict(list)
    for t in tasks:
        if t.zone_id is not None and t.sequence is not None:
            zone_dict[t.zone_id].append(t)

    for z_id, z_tasks in zone_dict.items():
        z_tasks_sorted = sorted(z_tasks, key=lambda x: x.sequence)
        for i in range(len(z_tasks_sorted) - 1):
            cur_t = z_tasks_sorted[i]
            nxt_t = z_tasks_sorted[i+1]
            cur_end = task_vars[cur_t.task_id][2]
            nxt_start = task_vars[nxt_t.task_id][0]
            model.Add(nxt_start >= cur_end)
    logger.info("Added sequence constraints.")
    # 6) Minimize makespan
    all_end_vars = [task_vars[t.task_id][2] for t in tasks]
    makespan = model.NewIntVar(0, HORIZON, "makespan")
    model.AddMaxEquality(makespan, all_end_vars)
    model.Minimize(makespan)
    logger.info("Minimizing makespan.")
    # ------------------------------------------------------------------
    # D) Solve the model
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_solver_time_s
    logger.info("Solving the model.")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.info("No feasible solution found.")
        return {}, None

    logger.info("Solver status: {}", solver.StatusName(status))

    # ------------------------------------------------------------------
    # E) Build final schedule output
    #    We'll provide:
    #      schedule["by_task"]
    #      schedule["by_resource"]
    # ------------------------------------------------------------------
    schedule = {
        "by_task": {},
        "by_resource": {}
    }

    # Initialize resource-lists
    for r in resources:
        schedule["by_resource"][r.resource_id] = []

    # Helper to map an integer offset back to a real datetime
    def int_to_dt(offset: int) -> datetime:
        if time_unit == "minute":
            return global_earliest + timedelta(minutes=offset)
        elif time_unit == "hour":
            return global_earliest + timedelta(hours=offset)
        else:
            raise ValueError("Unsupported time_unit. Use 'minute' or 'hour'.")

    # Fill "by_task"
    for t in tasks:
        (s_var, dur, e_var, _) = task_vars[t.task_id]
        start_val = solver.Value(s_var)
        end_val   = solver.Value(e_var)
        # map back to real-world dt
        real_start = int_to_dt(start_val)
        real_end   = int_to_dt(end_val)

        used_equip = []
        for eq in eq_list:
            if solver.Value(equip_used[(t.task_id, eq.resource_id)]) == 1:
                used_equip.append(eq.resource_id)

        used_workers = []
        for w in wr_list:
            if solver.Value(worker_used[(t.task_id, w.resource_id)]) == 1:
                used_workers.append(w.resource_id)

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
            "team": t.required_skill,
            "sequence": t.sequence,
            "zone_id": t.zone_id,
            "initial_start": t.start_earliest,
            "initial_end": t.end_latest
        }

    # Fill "by_resource"
    for t in tasks:
        (s_var, _, e_var, _) = task_vars[t.task_id]
        start_val = solver.Value(s_var)
        end_val   = solver.Value(e_var)
        real_start = int_to_dt(start_val)
        real_end   = int_to_dt(end_val)

        # check equip usage
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
                    "optional_count" : t.optional_equip_count,
                })

        # check worker usage
        for w in wr_list:
            if solver.Value(worker_used[(t.task_id, w.resource_id)]) == 1:
                schedule["by_resource"][w.resource_id].append({
                    "task_id": t.task_id,
                    "task_name": t.name,
                    "start_int": start_val,
                    "end_int": end_val,
                    "start_optimized": real_start,
                    "end_optimized": real_end,
                    "team": t.required_skill
                })

    # Sort each resource’s intervals by start_int for clarity
    for r_id, intervals in schedule["by_resource"].items():
        intervals.sort(key=lambda x: x["start_int"])

    makespan_val = solver.Value(makespan)
    logger.info("Schedule built successfully with makespan: {}", makespan_val)
    print("Makespan (int) =", makespan_val)
    print("Makespan start_dt => end_dt =",
          int_to_dt(0), "=>", int_to_dt(makespan_val))

    return schedule, makespan_val