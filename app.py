from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
from src.optimizer import OptimizedProductionScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'api.log',
            maxBytes=10000000,
            backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

# Pydantic Models
class DateRange(BaseModel):
    start_date: datetime = Field(..., description="Start date for optimization")
    end_date: datetime = Field(..., description="End date for optimization")
    scope: int = Field(26, description="Scope for optimization")

    # @field_validator('end_date')
    # def end_date_must_be_after_start_date(cls, v, values):
    #     if 'start_date' in values and v <= values['start_date']:
    #         raise ValueError('end_date must be after start_date')
    #     return v

    # @field_validator('start_date', 'end_date')
    # def dates_must_be_future(cls, v):
    #     if v < datetime.now():
    #         raise ValueError('dates must be in the future')
    #     return v

class Equipment(BaseModel):
    equipment_id: int
    equipment_name: str
    equipment_type: str
    category: str

class Worker(BaseModel):
    worker_id: int
    worker_name: str
    team_name: str

class Task(BaseModel):
    task_id: str
    name: str
    category: Optional[str]
    startDate: str
    endDate: str
    workload: int
    sequence: int
    required_workers: str
    mandatoryCount: int
    optionalCount: int
    equipments: Optional[Dict]
    num: int

class Process(BaseModel):
    process_id: int
    process_name: str
    tasks: List[Task]

class EquipmentUtilization(BaseModel):
    total_minutes: float
    equipment_name: str
    utilization_percentage: float

class Statistics(BaseModel):
    total_processes: int
    total_tasks: int
    makespan_minutes: int
    equipment_utilization: Dict[str, EquipmentUtilization]
    total_equipment_assignments: int

class OptimizationResponse(BaseModel):
    status: str
    solution: Optional[List[Process]]
    statistics: Optional[Statistics]
    optimization_time: Optional[float]

# Database configuration
DATABASE_URL = "mssql+pyodbc://oleksii:sdjnn4393vn@194.76.26.191/FLIPCLOUD?driver=ODBC+Driver+17+for+SQL+Server"

# Create FastAPI app with lifespan
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
    title="Production Schedule Optimizer",
    description="API for optimizing production schedules",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
async def get_db():
    db = app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get optimizer instance
async def get_optimizer(db=Depends(get_db)):
    return OptimizedProductionScheduler(DATABASE_URL)

@app.post("/optimize/", response_model=OptimizationResponse)
async def optimize_schedule(
    date_range: DateRange,
    optimizer: OptimizedProductionScheduler = Depends(get_optimizer)
) -> OptimizationResponse:
    """
    Optimize production schedule for the given date range.
    """
    logger.info(f"Received optimization request for period: {date_range.start_date} to {date_range.end_date}")
    
    try:
        result = optimizer.optimize_schedule(date_range.start_date, date_range.end_date, scope=date_range.scope)
        
        if result['status'] == 'no_processes':
            raise HTTPException(
                status_code=404,
                detail="No processes found within the specified date range"
            )
        
        if result['status'] == 'infeasible':
            raise HTTPException(
                status_code=200,
                detail="Processes are already optimized for the specified date range"
            )
        
        # Convert integer keys in equipment_utilization to strings
        if "statistics" in result and "equipment_utilization" in result["statistics"]:
            result["statistics"]["equipment_utilization"] = {
                str(key): value for key, value in result["statistics"]["equipment_utilization"].items()
            }
        
        logger.info(f"Optimization completed with status: {result['status']}")
        return OptimizationResponse(**result)
    
    except Exception as e:
        logger.exception("Optimization failed")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )

@app.get("/health/")
async def health_check():
    """
    Check API health status.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/version/")
async def version():
    """
    Get API version information.
    """
    return {
        "version": "1.0.0",
        "name": "Production Schedule Optimizer",
        "last_updated": "2024-12-11"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)