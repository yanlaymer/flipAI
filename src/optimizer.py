from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from ortools.sat.python import cp_model
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import sys

@dataclass
class ResourceData:
    """Container for precomputed resource information"""
    workers: Dict[int, Dict]
    equipment: Dict[int, Dict]
    worker_shifts: Dict[int, List[Tuple[datetime, datetime]]]

class OptimizedProductionScheduler:
    def __init__(self, connection_string: str, scope_id: int = 26):
        """Initialize scheduler with database connection and configuration."""
        self.engine = create_engine(connection_string, pool_size=20, max_overflow=10)
        self.scope_id = scope_id
        self.start_date = None
        self.end_date = None
        self.model = None
        self.resource_data = None
        self._setup_solver()
        self._setup_logging()

    def _setup_solver(self):
        """Initialize the CP-SAT solver with optimized parameters."""
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300
        self.solver.parameters.num_search_workers = 16
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.cp_model_presolve = False
        self.solver.parameters.linearization_level = 0
        self.solver.parameters.cp_model_probing_level = 0

    def _setup_logging(self):
        """Configure logging with Loguru."""
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        # logger.add(
        #     "scheduler_{time}.log",
        #     rotation="500 MB",
        #     retention="10 days",
        #     level="DEBUG"
        # )

    def _precompute_resources(self, start_date: datetime, end_date: datetime) -> ResourceData:
        """Precompute all resource data with a single optimized query."""
        query = text("""
        WITH WorkerData AS (
            SELECT DISTINCT
                p.PERS_PK as workerId,
                p.PERS_FIRSTNAME + ' ' + p.PERS_LASTNAME as workerName,
                p.PERS_FILTER3 as teamName,
                fs.SH_DATE as shiftDate,
                fs.SH_START as shiftStartMinutes,
                CASE 
                    WHEN fs.SH_STOP > fs.SH_START THEN fs.SH_STOP
                    ELSE fs.SH_STOP + 1440
                END as shiftEndMinutes
            FROM PERSONNEL p
            JOIN AGGREGATION a ON p.PERS_PK = a.AGG_PERS_FK
            LEFT JOIN FLIPSHIFT fs ON p.PERS_PK = fs.SH_PERS_FK
            WHERE a.AGG_PLAN_FK = :scope_id
        ),
        EquipmentData AS (
            SELECT 
                r.RE_PK as equipmentId,
                r.RE_NAME as equipmentName,
                r.RE_TYPE as equipmentType,
                tr.tr_name as category,
                r.re_maint_required as maintenanceRequired
            FROM RESSOURCE r
            JOIN typeressource tr ON r.re_resstype_fk = tr.tr_pk
            WHERE r.RE_TYPE = 'EQUIPEMENT'
        )
        SELECT 
            ISNULL((SELECT * FROM WorkerData FOR JSON PATH), '[]') as worker_data,
            ISNULL((SELECT * FROM EquipmentData FOR JSON PATH), '[]') as equipment_data
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"scope_id": self.scope_id}).first()
            
            if not result:
                logger.error("Failed to retrieve resource data")
                raise ValueError("No resource data available")

            worker_data = json.loads(result.worker_data)
            equipment_data = json.loads(result.equipment_data)
            
            logger.info(f"Retrieved {len(worker_data)} workers and {len(equipment_data)} equipment items")

            workers = {}
            worker_shifts = {}
            for wd in worker_data:
                worker_id = wd['workerId']
                if worker_id not in workers:
                    workers[worker_id] = {
                        'worker_id': worker_id,
                        'worker_name': f"{wd.get('PERS_FIRSTNAME', '')} {wd.get('PERS_LASTNAME', '')}",
                        'team_name': wd['teamName']
                    }
                    worker_shifts[worker_id] = []
                
                worker_shifts[worker_id].append((
                    wd['shiftStartMinutes'],
                    wd['shiftEndMinutes']
                ))

            equipment = {
                eq['equipmentId']: {
                    'equipment_id': eq['equipmentId'],
                    'equipment_name': eq['equipmentName'],
                    'equipment_type': eq['equipmentType'],
                    'category': eq['category']
                }
                for eq in equipment_data
            }

            return ResourceData(workers=workers, equipment=equipment, worker_shifts=worker_shifts)

    def get_processes_in_daterange(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Retrieve processes and tasks within the specified date range."""
        query = text("""
            SELECT 
                pa.pra_pro_fk as process_id,
                pr.PRO_NAME as process_name,
                va.act_id,
                va.act_name,
                va.act_category,
                va.act_workload,
                va.defaultStatus,
                va.workCadence,
                va.workersPerBox,
                va.prefweekend,
                va.block_zone,
                pa.pra_seq,
                pa.pra_groupnum,
                pa.pra_interval,
                pa.pra_max_interval,
                pa.pra_start_interval,
                pa.pra_workload,
                pa.PRA_PARA_ALIGN,
                ISNULL((
                    SELECT ar2.ACR_QTY, ar2.ACR_OPTION, r2.RE_NAME, r2.RE_TYPE
                    FROM actressource ar2
                    JOIN ressource r2 ON ar2.acr_ress_fk = r2.re_pk
                    WHERE va.act_id = ar2.acr_cm_fk AND r2.re_type = 'TEAM'
                    FOR JSON PATH
                ), '[]') as requiredWorkers,
                r.RE_TYPE as resource_type,
                r.RE_NAME as resource_name,
                r.RE_PK as resource_id,
                tr.tr_name as equipment_category,
                ar.ACR_QTY as resource_quantity,
                ISNULL(ar.ACR_OPTION, 'auto') as resource_option,
                ar.ACR_WORKLOAD as equip_workload,
                MIN(pl.pll_datestart) OVER (PARTITION BY pa.pra_pro_fk) as start_date,
                MAX(pl.pll_datestop) OVER (PARTITION BY pa.pra_pro_fk) as end_date
            FROM processactivity pa
            JOIN v_activity va ON pa.PRA_ACT_FK = va.act_id
            LEFT JOIN actressource ar ON va.act_id = ar.acr_cm_fk
            LEFT JOIN ressource r ON ar.acr_ress_fk = r.re_pk
            LEFT JOIN typeressource tr ON r.re_resstype_fk = tr.tr_pk
            JOIN planningline pl ON pa.pra_pro_fk = pl.pll_pro_fk
            JOIN process pr ON pa.pra_pro_fk = pr.PRO_PK
            WHERE va.act_scope = :scope_id
            AND pl.pll_datestart >= :start_date
            AND pl.pll_datestop <= :end_date
            ORDER BY pa.pra_seq
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {
                "scope_id": self.scope_id,
                "start_date": start_date,
                "end_date": end_date
            })

            processes = {}
            for row in result.mappings():
                process_id = row['process_id']
                task_id = row['act_id']
                equipment_category = row['equipment_category']
                
                if process_id not in processes:
                    processes[process_id] = {
                        'process_id': process_id,
                        'process_name': row['process_name'],
                        'start_date': row['start_date'],
                        'end_date': row['end_date'],
                        'tasks': []
                    }

                task_key = f"{task_id}_{row['process_name']}_{row['pra_seq']}"
                if equipment_category:
                    task_key += f"_{equipment_category}"

                existing_task = next(
                    (t for t in processes[process_id]['tasks'] if t['id'] == task_key),
                    None
                )

                if not existing_task:
                    new_task = {
                        'id': task_key,
                        'pk': task_id,
                        'name': row['act_name'],
                        'type': row['act_category'],
                        'num': row['pra_seq'],
                        'equipCategory': equipment_category,
                        'workload': row['pra_workload'] or row['act_workload'],
                        'defaultStatus': row['defaultStatus'],
                        'workCadence': row['workCadence'],
                        'predecessor': row['pra_groupnum'],
                        'interval': row['pra_interval'],
                        'maxInterval': row['pra_max_interval'],
                        'intervalStart': row['pra_start_interval'],
                        'required_workers': row['requiredWorkers'],
                        'equipments': [],
                        'mandatoryCount': 0,
                        'optionalCount': 0
                    }
                    
                    if row['resource_type'] == 'EQUIPEMENT':
                        new_task['equipments'].append(row['resource_id'])
                        if row['resource_option'] == 'required':
                            new_task['mandatoryCount'] += 1
                        elif row['resource_option'] == 'auto':
                            new_task['optionalCount'] += 1
                    
                    processes[process_id]['tasks'].append(new_task)
                else:
                    if row['resource_type'] == 'EQUIPEMENT':
                        existing_task['equipments'].append(row['resource_id'])
                        if row['resource_option'] == 'required':
                            existing_task['mandatoryCount'] += 1
                        elif row['resource_option'] == 'auto':
                            existing_task['optionalCount'] += 1

            return list(processes.values())

    def _sort_tasks_by_priority(self, tasks: List[Tuple[Dict, Dict]]) -> List[Tuple[Dict, Dict]]:
        """Sort tasks by priority based on multiple criteria."""
        def get_task_priority(task_tuple):
            task, _ = task_tuple
            priority = 0
            
            # Priority based on sequence number
            priority += 1000 - (task.get('num', 0) or 0)
            
            # Priority based on mandatory equipment
            priority += task.get('mandatoryCount', 0) * 100
            
            # Priority based on required workers
            try:
                required_workers = json.loads(task.get('required_workers', '[]'))
                priority += sum(w.get('ACR_QTY', 0) for w in required_workers) * 50
            except json.JSONDecodeError:
                pass
            
            # Priority based on workload
            priority += task.get('workload', 0) * 0.1
            
            return priority

        return sorted(tasks, key=get_task_priority, reverse=True)

    def _create_process_variables(self, process: Dict, horizon: int) -> Dict:
        """Create variables for tasks and resources with proper initialization."""
        variables = {
            'tasks': {},
            'equipment': {},  # Initialize equipment dict at top level
            'workers': {}     # Initialize workers dict at top level
        }

        sorted_tasks = sorted(
            process['tasks'],
            key=lambda x: x.get('num', float('inf'))
        )

        for task in sorted_tasks:
            task_id = task['id']
            duration = task['workload']

            # Create task timing variables
            start_var = self.model.NewIntVar(0, horizon - duration, f'start_{task_id}')
            end_var = self.model.NewIntVar(duration, horizon, f'end_{task_id}')
            interval_var = self.model.NewIntervalVar(
                start_var, duration, end_var, f'interval_{task_id}'
            )

            variables['tasks'][task_id] = {
                'start': start_var,
                'duration': duration,
                'end': end_var,
                'interval': interval_var
            }

            # Create equipment variables if category exists
            if task.get('equipCategory'):
                equipment_vars = {}
                relevant_equipment = [
                    (eq_id, eq_info) 
                    for eq_id, eq_info in self.resource_data.equipment.items()
                    if eq_info['category'] == task['equipCategory']
                ]
                
                if relevant_equipment:
                    for eq_id, eq_info in relevant_equipment:
                        equipment_vars[eq_id] = self.model.NewBoolVar(
                            f'equipment_{task_id}_{eq_id}'
                        )
                    variables['equipment'][task_id] = equipment_vars

        return variables

    def _add_group_optimization_constraints(self, group: List[Dict], group_vars: Dict):
        """Add constraints to optimize resource usage within a group."""
        all_tasks = []
        for process in group:
            process_vars = group_vars[process['process_id']]
            for task in process['tasks']:
                if task['id'] in process_vars['tasks']:
                    task_vars = process_vars['tasks'][task['id']]
                    all_tasks.append((task, task_vars))

        sorted_tasks = self._sort_tasks_by_priority(all_tasks)

        # Add resource capacity constraints
        self._add_resource_capacity_constraints(sorted_tasks, group_vars)

        # Add task timing constraints
        for i in range(len(sorted_tasks) - 1):
            current_task, current_vars = sorted_tasks[i]
            next_task, next_vars = sorted_tasks[i + 1]
            
            # Add transition time if tasks share resources
            if self._share_resources(current_task, next_task):
                transition_time = self.model.NewIntVar(
                    10,  # Minimum transition time
                    30,  # Maximum transition time
                    f'transition_{current_task["id"]}_{next_task["id"]}'
                )
                self.model.Add(next_vars['start'] >= current_vars['end'] + transition_time)

    def _share_resources(self, task1: Dict, task2: Dict) -> bool:
        """Determine if two tasks share any resources."""
        # Check equipment category overlap
        if task1.get('equipCategory') and task1['equipCategory'] == task2.get('equipCategory'):
            return True

        # Check worker team overlap
        try:
            workers1 = json.loads(task1.get('required_workers', '[]'))
            workers2 = json.loads(task2.get('required_workers', '[]'))
            teams1 = {w['RE_NAME'] for w in workers1}
            teams2 = {w['RE_NAME'] for w in workers2}
            if teams1 & teams2:
                return True
        except json.JSONDecodeError:
            pass

        return False

    def _add_resource_capacity_constraints(self, sorted_tasks: List[Tuple[Dict, Dict]], group_vars: Dict):
        """Add constraints for resource capacity and availability."""
        # Group tasks by equipment category
        category_tasks = {}
        for task, task_vars in sorted_tasks:
            category = task.get('equipCategory')
            if category:
                if category not in category_tasks:
                    category_tasks[category] = []
                category_tasks[category].append((task, task_vars))

        # Add constraints for each equipment category
        for category, tasks in category_tasks.items():
            intervals = []
            for task, task_vars in tasks:
                interval = task_vars.get('interval')
                if interval:
                    intervals.append(interval)

            # Add no-overlap constraint for tasks using the same equipment category
            if intervals:
                self.model.AddNoOverlap(intervals)

    def _group_related_processes(self, processes: List[Dict]) -> List[List[Dict]]:
        """Group related processes based on shared resources and dependencies."""
        groups = []
        unassigned = set(range(len(processes)))

        while unassigned:
            current_group = []
            process_queue = [processes[unassigned.pop()]]

            while process_queue:
                current_process = process_queue.pop(0)
                current_group.append(current_process)

                # Find related processes
                for idx in list(unassigned):
                    other_process = processes[idx]
                    if self._are_processes_related(current_process, other_process):
                        process_queue.append(other_process)
                        unassigned.remove(idx)

            groups.append(current_group)

        return groups

    def _are_processes_related(self, process1: Dict, process2: Dict) -> bool:
        """Determine if processes are related based on shared resources or timing."""
        # Check for shared equipment categories
        categories1 = {
            task.get('equipCategory')
            for task in process1['tasks']
            if task.get('equipCategory')
        }
        categories2 = {
            task.get('equipCategory')
            for task in process2['tasks']
            if task.get('equipCategory')
        }
        if categories1 & categories2:
            return True

        # Check for shared worker teams
        teams1 = set()
        teams2 = set()
        for task in process1['tasks']:
            try:
                workers = json.loads(task.get('required_workers', '[]'))
                teams1.update(w['RE_NAME'] for w in workers)
            except json.JSONDecodeError:
                pass
        
        for task in process2['tasks']:
            try:
                workers = json.loads(task.get('required_workers', '[]'))
                teams2.update(w['RE_NAME'] for w in workers)
            except json.JSONDecodeError:
                pass

        if teams1 & teams2:
            return True

        return False

    def optimize_schedule(self, start_date: datetime, end_date: datetime, scope:int = 26) -> Dict:
        """Main optimization function to generate the schedule."""
        logger.info("Starting optimization for period: {} to {}", start_date, end_date)
        optimization_start = datetime.now()
        self.scope = scope

        try:
            self.start_date = start_date
            self.end_date = end_date

            # Precompute resources and get processes
            self.resource_data = self._precompute_resources(start_date, end_date)
            processes = self.get_processes_in_daterange(start_date, end_date)

            if not processes:
                logger.error("No processes found within the date range")
                return {'status': 'no_processes'}

            # Create and solve the optimization model
            model, variables = self._create_improved_optimization_model(processes)
            status = self._solve_model(model)

            optimization_time = (datetime.now() - optimization_start).total_seconds()
            logger.info("Optimization completed in {:.2f} seconds", optimization_time)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                solution = self._build_solution(variables, processes)
                return {
                    'status': 'optimal' if status == cp_model.OPTIMAL else 'feasible',
                    'solution': solution,
                    'statistics': self._calculate_statistics(solution),
                    'optimization_time': optimization_time
                }

            return {'status': 'infeasible'}

        except Exception as e:
            logger.exception("Optimization failed: {}", str(e))
            raise

    def _solve_model(self, model: cp_model.CpModel) -> int:
        """Solve the optimization model with progress tracking."""
        class SolutionCallback(cp_model.CpSolverSolutionCallback):
            def __init__(self):
                super().__init__()
                self._solutions = 0
                self._start_time = datetime.now()

            def on_solution_callback(self):
                self._solutions += 1
                elapsed = (datetime.now() - self._start_time).total_seconds()
                logger.info(
                    "Solution {} found after {:.2f}s with objective: {}",
                    self._solutions, elapsed, self.ObjectiveValue()
                )

        callback = SolutionCallback()
        status = self.solver.Solve(model, callback)
        return status

    def _build_solution(self, variables: Dict, processes: List[Dict]) -> List[Dict]:
        """Build the final schedule solution."""
        solution = []
        for process in processes:
            process_tasks = []
            process_vars = variables[process['process_id']]

            for task in process['tasks']:
                task_id = task['id']
                task_vars = process_vars['tasks'][task_id]

                # Calculate task timing
                start_time = self.start_date + timedelta(
                    minutes=self.solver.Value(task_vars['start'])
                )
                end_time = self.start_date + timedelta(
                    minutes=self.solver.Value(task_vars['end'])
                )

                # Get assigned equipment
                assigned_equipment = None
                if task_id in process_vars.get('equipment', {}):
                    for eq_id, var in process_vars['equipment'][task_id].items():
                        if self.solver.Value(var) == 1:
                            eq_info = self.resource_data.equipment[eq_id]
                            assigned_equipment = {
                                'equipment_id': eq_id,
                                'equipment_name': eq_info['equipment_name'],
                                'equipment_type': eq_info['equipment_type'],
                                'category': eq_info['category']
                            }
                            break

                # Create task solution
                process_tasks.append({
                    'task_id': task_id,
                    'name': task['name'],
                    'category': task['equipCategory'],
                    'startDate': start_time.isoformat(),
                    'endDate': end_time.isoformat(),
                    'workload': task['workload'],
                    'sequence': task['num'],
                    'required_workers': task['required_workers'],
                    'mandatoryCount': task.get('mandatoryCount', 0),
                    'optionalCount': task.get('optionalCount', 0),
                    'equipments': assigned_equipment,
                    'num': task['num']
                })

            solution.append({
                'process_id': process['process_id'],
                'process_name': process['process_name'],
                'tasks': sorted(process_tasks, key=lambda x: x['sequence'])
            })

        return solution

    def _calculate_statistics(self, solution: List[Dict]) -> Dict:
        """Calculate statistics about the generated schedule."""
        stats = {
            'total_processes': len(solution),
            'total_tasks': 0,
            'makespan_minutes': 0,
            'equipment_utilization': {},
            'total_equipment_assignments': 0
        }

        earliest_start = None
        latest_end = None

        for process in solution:
            for task in process['tasks']:
                stats['total_tasks'] += 1
                
                task_start = datetime.fromisoformat(task['startDate'])
                task_end = datetime.fromisoformat(task['endDate'])

                if earliest_start is None or task_start < earliest_start:
                    earliest_start = task_start
                if latest_end is None or task_end > latest_end:
                    latest_end = task_end

                if task['equipments']:
                    stats['total_equipment_assignments'] += 1
                    eq_id = task['equipments']['equipment_id']
                    if eq_id not in stats['equipment_utilization']:
                        stats['equipment_utilization'][eq_id] = {
                            'total_minutes': 0,
                            'equipment_name': task['equipments']['equipment_name']
                        }
                    stats['equipment_utilization'][eq_id]['total_minutes'] += (
                        task_end - task_start
                    ).total_seconds() / 60

        if earliest_start and latest_end:
            stats['makespan_minutes'] = int(
                (latest_end - earliest_start).total_seconds() / 60
            )

        # Calculate equipment utilization percentages
        total_horizon = (self.end_date - self.start_date).total_seconds() / 60
        for eq_id in stats['equipment_utilization']:
            stats['equipment_utilization'][eq_id]['utilization_percentage'] = (
                (stats['equipment_utilization'][eq_id]['total_minutes'] / total_horizon) * 100
            )

        return stats
    
    def _add_process_constraints(self, process: Dict, variables: Dict):
        """Add constraints for task sequencing and resource allocation within a process."""
        # Sort tasks by sequence number
        sorted_tasks = sorted(
            process['tasks'],
            key=lambda x: x.get('num', float('inf'))
        )

        # Add sequence constraints
        for i in range(len(sorted_tasks) - 1):
            current_task = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            current_vars = variables['tasks'][current_task['id']]
            next_vars = variables['tasks'][next_task['id']]
            
            # Ensure tasks follow sequence
            self.model.Add(next_vars['start'] >= current_vars['end'])

        # Add equipment constraints for each task
        for task in sorted_tasks:
            task_id = task['id']
            logger.info(f"Task {task_id}: mandatoryCount={task.get('mandatoryCount', 0)}, optionalCount={task.get('optionalCount', 0)}")
            
            # Only process equipment constraints if the task has equipment requirements
            if task.get('equipCategory') and task_id in variables.get('equipment', {}):
                equipment_vars = variables['equipment'][task_id]
                
                # Add mandatory equipment constraints
                mandatory_count = task.get('mandatoryCount', 0)
                if mandatory_count > 0:
                    self.model.Add(sum(equipment_vars.values()) >= mandatory_count)
                else:
                    logger.info(f"No mandatory equipment for task {task_id}")
                
                # Add optional equipment constraints if specified
                optional_count = task.get('optionalCount', 0)
                if optional_count > 0:
                    self.model.Add(sum(equipment_vars.values()) <= mandatory_count + 1)
    def _create_improved_optimization_model(self, processes: List[Dict]) -> Tuple[cp_model.CpModel, Dict]:
        """Create an improved optimization model with more flexible constraints."""
        self.model = cp_model.CpModel()
        
        # Calculate horizon with more flexibility
        total_workload = sum(
            task['workload'] 
            for process in processes 
            for task in process['tasks']
        )
        
        # Add larger buffer for flexibility
        horizon = int(max(
            total_workload * 3,  # Triple the workload for more flexibility
            int((self.end_date - self.start_date).total_seconds() / 60)
        ))
        
        variables = {}
        end_times = []
        
        # Process tasks chronologically
        for process in processes:
            process_vars = self._create_process_variables(process, horizon)
            variables[process['process_id']] = process_vars
            
            # Add basic process constraints with relaxed timing
            self._add_flexible_process_constraints(process, process_vars)
            
            # Collect end times for makespan calculation
            for task in process['tasks']:
                task_vars = process_vars['tasks'][task['id']]
                end_times.append(task_vars['end'])
        
        # Add global resource constraints with flexibility
        self._add_flexible_resource_constraints(processes, variables)
        
        # Set objective to minimize makespan
        makespan = self.model.NewIntVar(0, horizon, 'makespan')
        self.model.AddMaxEquality(makespan, end_times)
        self.model.Minimize(makespan)
        
        return self.model, variables

    def _add_flexible_process_constraints(self, process: Dict, variables: Dict):
        """Add process constraints with more flexibility."""
        sorted_tasks = sorted(process['tasks'], key=lambda x: x.get('num', float('inf')))
        
        # Add sequence constraints between consecutive tasks
        for i in range(len(sorted_tasks) - 1):
            current_task = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            current_vars = variables['tasks'][current_task['id']]
            next_vars = variables['tasks'][next_task['id']]
            
            # Allow some flexibility in task ordering
            min_gap = 0  # Minimum gap between tasks
            if self._share_resources(current_task, next_task):
                min_gap = 10  # Minimum transition time if sharing resources
            
            self.model.Add(next_vars['start'] >= current_vars['end'] + min_gap)
        
        # Handle equipment constraints
        for task in sorted_tasks:
            task_id = task['id']
            logger.info(f"Task {task_id}: mandatoryCount={task.get('mandatoryCount', 0)}, "
                    f"optionalCount={task.get('optionalCount', 0)}")
            
            if task_id in variables.get('equipment', {}):
                equipment_vars = variables['equipment'][task_id]
                
                # Mandatory equipment constraints
                mandatory_count = task.get('mandatoryCount', 0)
                if mandatory_count > 0:
                    # Allow slightly fewer mandatory equipment if needed
                    min_mandatory = max(1, mandatory_count - 1)
                    self.model.Add(sum(equipment_vars.values()) >= min_mandatory)
                else:
                    logger.info(f"No mandatory equipment for task {task_id}")

    def _add_flexible_resource_constraints(self, processes: List[Dict], variables: Dict):
        """Add global resource constraints with flexibility."""
        # Group tasks by equipment category
        category_tasks = {}
        
        for process in processes:
            process_vars = variables[process['process_id']]
            for task in process['tasks']:
                category = task.get('equipCategory')
                if category:
                    if category not in category_tasks:
                        category_tasks[category] = []
                    task_vars = process_vars['tasks'][task['id']]
                    category_tasks[category].append((task, task_vars))
        
        # Add constraints for each equipment category
        for category, tasks in category_tasks.items():
            relevant_equipment = [
                eq_id for eq_id, eq_info in self.resource_data.equipment.items()
                if eq_info['category'] == category
            ]
            
            if not relevant_equipment:
                continue
            
            # Create intervals for tasks using this equipment category
            intervals = []
            for task, task_vars in tasks:
                interval = self.model.NewIntervalVar(
                    task_vars['start'],
                    task_vars['duration'],
                    task_vars['end'],
                    f'interval_{task["id"]}'
                )
                intervals.append(interval)
            
            # Allow some overlap between tasks if needed
            if intervals:
                # Use cumulative constraint instead of no-overlap
                max_concurrent = len(relevant_equipment)
                demands = [1] * len(intervals)  # Each task requires one unit of resource
                self.model.AddCumulative(intervals, demands, max_concurrent)

    def _share_resources(self, task1: Dict, task2: Dict) -> bool:
        """Determine if tasks share resources, with more flexible matching."""
        # Check equipment category overlap
        if (task1.get('equipCategory') and task2.get('equipCategory') and
                task1['equipCategory'] == task2['equipCategory']):
            return True
        
        # Check worker team overlap
        try:
            workers1 = json.loads(task1.get('required_workers', '[]'))
            workers2 = json.loads(task2.get('required_workers', '[]'))
            
            teams1 = {w['RE_NAME'] for w in workers1}
            teams2 = {w['RE_NAME'] for w in workers2}
            
            if teams1 & teams2:  # Check for any overlap in teams
                return True
        except json.JSONDecodeError:
            pass
        
        return False
                        
    def _check_solution_feasibility(self, solution: List[Dict]) -> List[str]:
        """Verify solution feasibility and constraints, such as worker availability and overlapping assignments."""
        issues = []

        # Check for worker assignment conflicts (overlapping tasks)
        worker_assignments = {}
        for process in solution:
            for task in process['tasks']:
                task_start = datetime.fromisoformat(task['startDate'])
                task_end = datetime.fromisoformat(task['endDate'])

                for worker in task['assignedWorkers']:
                    worker_id = worker['worker_id']
                    if worker_id not in worker_assignments:
                        worker_assignments[worker_id] = []
                    
                    # Check for overlapping assignments for the worker
                    for existing_task_start, existing_task_end in worker_assignments[worker_id]:
                        if task_start < existing_task_end and task_end > existing_task_start:
                            issues.append(
                                f"Worker {worker['worker_name']} has overlapping "
                                f"assignments between {task_start} and {task_end}"
                            )
                    
                    worker_assignments[worker_id].append((task_start, task_end))

                # Check if workers are assigned to tasks outside their shifts
                for worker in task['assignedWorkers']:
                    worker_id = worker['worker_id']
                    if not self._validate_worker_availability(worker_id, task_start, task_end):
                        issues.append(
                            f"Worker {worker['worker_name']} assigned outside "
                            f"their shift for task starting at {task_start}"
                        )

        return issues