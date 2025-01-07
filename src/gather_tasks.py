from sqlalchemy import text
from datetime import datetime
from loguru import logger

def gather_all_scheduling_data(conn, scope_id, start_date, end_date, max_tasks=50):
    """
    Gathers core scheduling data for an optimization:
      1) Processes/Tasks within [start_date, end_date].
      2) Equipment from RESSOURCE/typeressource.
      3) Workers + shifts from PERSONNEL/FLIPSHIFT.

    Returns a dict:
    {
      "processes": [ {process/task info}, ... ],
      "equipment": [ {equipment info}, ... ],
      "workers": { worker_id: { 'firstName':..., 'team_name':...}, ... },
      "shifts": { worker_id: [(shift_start, shift_end, shift_date), ...], ... }
    }

    - scope_id: Your scope integer
    - start_date, end_date: Python datetime objects
    - max_tasks: limit tasks for testing (defaults to 50)
    """
    
    logger.info("Gathering scheduling data for scope_id: {}, start_date: {}, end_date: {}", scope_id, start_date, end_date)

    # 1) Processes in date range
    #    We join PROCESS + PROCESSACTIVITY + COREMATRIX + planningLine
    logger.info("Executing process query.")
    process_query = text(f"""
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

    # 2) Equipment data
    logger.info("Executing equipment query.")
    equipment_query = text("""
        SELECT
            r.RE_PK as equipment_id,
            r.RE_NAME as equipment_name,
            r.RE_TYPE as equipment_type,
            tr.tr_name as category,
            r.re_maint_required as maintenanceRequired,
            r.re_parent as parent_zone
        FROM RESSOURCE r
        JOIN typeressource tr ON r.re_resstype_fk = tr.tr_pk
        WHERE r.RE_SCOPE_FK = :scope_id
          AND r.RE_TYPE = 'EQUIPEMENT'
    """)
    
    logger.info("Executing zone query.")
    zone_query = text("""
        SELECT
            r.RE_PK as zone_id,
            r.RE_NAME as zone_name,
            r.RE_TYPE as zone_type,
            tr.tr_name as category,
            r.re_maint_required as maintenanceRequired,
            r.re_parent as parent_zone
        FROM RESSOURCE r
        JOIN typeressource tr ON r.re_resstype_fk = tr.tr_pk
        WHERE r.RE_SCOPE_FK = :scope_id
          AND r.RE_TYPE = 'ZONE'
    """)

    # 3) Workers + shifts
    logger.info("Executing workers query.")
    workers_query = text("""
        SELECT
          p.PERS_PK as worker_id,
          p.PERS_FIRSTNAME as firstName,
          p.PERS_LASTNAME  as lastName,
          p.PERS_FILTER3   as team_name
        FROM PERSONNEL p
        JOIN AGGREGATION a ON a.AGG_PERS_FK = p.PERS_PK
        WHERE a.agg_plan_fk = :scope_id
    """)

    logger.info("Executing shifts query.")
    shifts_query = text("""
        SELECT
          fs.SH_PERS_FK as worker_id,
          fs.SH_DATE as shift_date,
          fs.SH_START as shift_start,
          CASE WHEN fs.SH_STOP > fs.SH_START THEN fs.SH_STOP
               ELSE fs.SH_STOP + 1440
          END as shift_end
        FROM FLIPSHIFT fs
        WHERE fs.sh_scope_fk = :scope_id
    """)

    # Execute the queries
    # Use the provided connection directly
    # Processes
    proc_result = conn.execute(
        process_query,
        {"scope_id": scope_id, "start_date": start_date, "end_date": end_date}
    ).mappings().all()
    processes_data = [dict(r) for r in proc_result]

    # Equipment
    equip_result = conn.execute(
        equipment_query, {"scope_id": scope_id}
    ).mappings().all()
    equipment_data = [dict(r) for r in equip_result]
    
    zone_result = conn.execute(
        zone_query, {"scope_id": scope_id}
    ).mappings().all()
    zones = [dict(r) for r in zone_result]

    # Workers
    worker_result = conn.execute(
        workers_query, {"scope_id": scope_id}
    ).mappings().all()

    # Convert to dict { worker_id: {...} }
    workers_data = {}
    for w in worker_result:
        wdict = dict(w)
        wid = wdict["worker_id"]
        workers_data[wid] = {
            "firstName": wdict["firstName"],
            "lastName":  wdict["lastName"],
            "team_name": wdict["team_name"]
        }

    # Shifts
    shift_result = conn.execute(
        shifts_query, {"scope_id": scope_id}
    ).mappings().all()

    # Convert to dict { worker_id: [ (shift_start, shift_end, shift_date), ... ] }
    shifts_map = {}
    for s in shift_result:
        sdict = dict(s)
        wid   = sdict["worker_id"]
        if wid not in shifts_map:
            shifts_map[wid] = []
        shifts_map[wid].append((
            sdict["shift_start"],
            sdict["shift_end"],
            sdict["shift_date"]
        ))

    logger.info("Data gathering complete.")
    # Return everything in one big dictionary
    return {
        "processes": processes_data,
        "equipment": equipment_data,
        "zones" : zones,
        "workers":   workers_data,
        "shifts":    shifts_map
    }
