# tasks_fetcher.py

from sqlalchemy import text
from typing import List
from datetime import datetime
from src.data_scheme import Task

from sqlalchemy import text
from datetime import datetime
from typing import List
from src.data_scheme import Task

def get_tasks_by_process(conn, process_id: str) -> List[Task]:
    """
    Fetch tasks with correct resource counts by summing acr_qty
    for mandatory and optional equipment and workers.
    """
    query = text("""
WITH task_resources AS (
    -- Pre-aggregate resource counts to avoid repeated subqueries
    SELECT 
        acr_cm_fk as task_id,
        SUM(CASE WHEN r.RE_TYPE = 'EQUIPEMENT' THEN ar.acr_qty ELSE 0 END) as mandatory_equip_count,
        SUM(CASE WHEN r.RE_TYPE = 'EQUIPEMENT' THEN ar.acr_qty ELSE 0 END) as optional_equip_count,
        SUM(CASE WHEN r.RE_TYPE = 'TEAM' THEN ar.acr_qty ELSE 0 END) as required_workers
    FROM actressource ar
    JOIN ressource r ON ar.acr_ress_fk = r.RE_PK
    GROUP BY acr_cm_fk
),
equipment_categories AS (
    -- Get equipment category using ROW_NUMBER instead of TOP 1
    SELECT DISTINCT
        ar.acr_cm_fk as task_id,
        FIRST_VALUE(tr.tr_name) OVER (PARTITION BY ar.acr_cm_fk ORDER BY tr.tr_pk) as equipment_category
    FROM actressource ar
    JOIN ressource eq ON ar.acr_ress_fk = eq.RE_PK
    JOIN typeressource tr ON eq.re_resstype_fk = tr.tr_pk
    WHERE eq.RE_TYPE = 'EQUIPEMENT'
),
predecessor_list AS (
    -- Predecessor calculation
    SELECT 
        pred.pra_pro_fk,
        pred.pra_seq,
        STRING_AGG(pred2.pra_act_fk, ',') as predecessor_ids
    FROM processactivity pred
    LEFT JOIN processactivity pred2 ON pred2.pra_pro_fk = pred.pra_pro_fk 
        AND pred2.pra_seq < pred.pra_seq
    GROUP BY pred.pra_pro_fk, pred.pra_seq
)
SELECT
    pa.pra_pro_fk AS process_id,
    pr.PRO_NAME AS process_name,
    va.act_id AS task_id,
    va.act_name AS task_name,
    va.act_workload AS workload,
    MIN(pl.pll_datestart) AS start_earliest,
    MAX(pl.pll_datestop) AS end_latest,
    ec.equipment_category,
    COALESCE(tr.mandatory_equip_count, 0) as mandatory_equip_count,
    COALESCE(tr.optional_equip_count, 0) as optional_equip_count,
    COALESCE(tr.required_workers, 0) as required_workers,
    pl2.predecessor_ids,
    rz.RE_PK AS zone_id,
    pa.pra_seq AS sequence
FROM processactivity pa
JOIN v_activity va ON pa.PRA_ACT_FK = va.act_id
JOIN process pr ON pa.pra_pro_fk = pr.PRO_PK
JOIN planningline pl ON pa.pra_pro_fk = pl.pll_pro_fk
LEFT JOIN ressource rz ON va.block_zone = rz.RE_NAME
LEFT JOIN task_resources tr ON va.act_id = tr.task_id
LEFT JOIN equipment_categories ec ON va.act_id = ec.task_id
LEFT JOIN predecessor_list pl2 ON pl2.pra_pro_fk = pa.pra_pro_fk AND pl2.pra_seq = pa.pra_seq
WHERE pa.pra_pro_fk = :process_id
GROUP BY
    pa.pra_pro_fk, pr.PRO_NAME,
    va.act_id, va.act_name, va.act_workload,
    rz.RE_PK, pa.pra_seq,
    ec.equipment_category,
    tr.mandatory_equip_count, tr.optional_equip_count, tr.required_workers,
    pl2.predecessor_ids
ORDER BY pa.pra_seq
    """)

    result = conn.execute(query, {"process_id": process_id})
    tasks = []
    for row in result.mappings():
        tasks.append(
            Task(
                task_id=row["task_id"],
                process_id=row["process_id"],
                process_name=row["process_name"],
                name=row["task_name"],
                workload=row["workload"] or 0,
                start_earliest=row["start_earliest"],
                end_latest=row["end_latest"],
                category_equip=row["equipment_category"] or "UNDEFINED",
                mandatory_equip_count=row["mandatory_equip_count"] or 0,
                optional_equip_count=row["optional_equip_count"] or 0,
                required_workers=row["required_workers"] or 0,
                predecessor_ids=(row["predecessor_ids"].split(',') 
                                 if row["predecessor_ids"] else []),
                zone_id=row["zone_id"],
                sequence=row["sequence"]
            )
        )
    return tasks


# resources_fetcher.py

from sqlalchemy import text
from datetime import datetime
from typing import List
from src.data_scheme import Resource
from loguru import logger

def get_all_resources(conn, scope_id: int) -> List[Resource]:
    """
    Fetch all equipment, zones, and workers for a given scope.
    Include SHIFT data for workers if available.
    """
    logger.info(f"Fetching resources for scope {scope_id}")

    # 1) Equipment + Zones
    equipment_query = text("""
        SELECT
            re.RE_PK AS resource_id,
            re.RE_NAME AS name,
            CASE 
                WHEN re.RE_TYPE = 'EQUIPEMENT' THEN 'EQUIPMENT'
                WHEN re.RE_TYPE = 'ZONE' THEN 'ZONE'
            END AS resource_type,
            tr.tr_name AS category,
            re.re_maint_required AS maintenance_required
        FROM ressource re
        JOIN typeressource tr ON re.re_resstype_fk = tr.tr_pk
        WHERE re.RE_SCOPE_FK = :scope_id
          AND re.RE_TYPE IN ('EQUIPEMENT', 'ZONE')
    """)

    # 2) Workers
    workers_query = text("""
        SELECT
            p.PERS_PK AS resource_id,
            CONCAT(p.PERS_FIRSTNAME, ' ', p.PERS_LASTNAME) AS name,
            'WORKER' AS resource_type,
            COALESCE(p.PERS_FILTER3, '') AS skills,
            fs.shifts
        FROM PERSONNEL p
        LEFT JOIN (
            SELECT
                SH_PERS_FK AS worker_id,
                STRING_AGG(
                    CONCAT(
                        SH_DATE, '|',
                        SH_START, '|',
                        CASE WHEN SH_STOP > SH_START THEN SH_STOP ELSE SH_STOP + 1440 END
                    ), ','
                ) AS shifts
            FROM FLIPSHIFT
            WHERE sh_scope_fk = :scope_id
            GROUP BY SH_PERS_FK
        ) fs ON p.PERS_PK = fs.worker_id
        WHERE EXISTS (
            SELECT 1 FROM AGGREGATION a 
            WHERE a.agg_pers_fk = p.PERS_PK
              AND a.agg_plan_fk = :scope_id
        )
    """)

    resources = []

    # Fetch EQUIPMENT + ZONES
    eq_result = conn.execute(equipment_query, {"scope_id": scope_id}).mappings()
    for row in eq_result:
        resources.append(Resource(
            resource_id=row["resource_id"],
            name=row["name"],
            resource_type=row["resource_type"],
            equipment_category=row["category"],
            capacity=1  # default capacity for equipment / zone
        ))

    # Fetch WORKERS + shifts
    wr_result = conn.execute(workers_query, {"scope_id": scope_id}).mappings()
    for row in wr_result:
        shifts = []
        if row["shifts"]:
            # Each worker's shifts is a comma-separated string e.g. "2025-01-05|480|960,2025-01-06|480|960"
            for shift_str in row["shifts"].split(','):
                try:
                    date_part, start_part, end_part = shift_str.split('|')
                    start_int = int(start_part)
                    end_int   = int(end_part)
                    shift_date = datetime.strptime(date_part, "%Y-%m-%d").date()
                    shifts.append((start_int, end_int, shift_date))
                except Exception as e:
                    logger.error(f"Error parsing shift {shift_str}: {str(e)}")

        resources.append(Resource(
            resource_id=row["resource_id"],
            name=row["name"],
            resource_type="WORKER",
            skills=[s.strip() for s in row["skills"].split(',') if s.strip()],
            shifts=shifts,
            capacity=1
        ))

    return resources
