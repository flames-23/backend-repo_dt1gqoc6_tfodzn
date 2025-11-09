from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class Dataset(BaseModel):
    name: str = Field(..., description="Original filename or dataset name")
    description: Optional[str] = Field(None, description="Optional description")
    columns: List[str] = Field(default_factory=list, description="Detected columns")
    row_count: int = 0

    class Config:
        arbitrary_types_allowed = True


class UploadResponse(BaseModel):
    dataset_id: str
    name: str
    columns: List[str]
    row_count: int


class AnalysisRequest(BaseModel):
    dataset_id: str
    x: Optional[str] = None
    y: Optional[str] = None
    group_by: Optional[str] = None
    limit: int = 1000


class ChartData(BaseModel):
    title: str
    type: str  # e.g., bar, line, pie
    labels: List[str]
    values: List[float]
    meta: Dict[str, Any] = {}


class DashboardResponse(BaseModel):
    charts: List[ChartData]
