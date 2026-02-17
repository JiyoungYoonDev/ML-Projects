from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal, Any

ModelName = Literal['decision_tree', 'random_forest', 'logistic_regression']

class KFoldConfig(BaseModel):
    n_splits: int = 10
    shuffle: bool = True
    random_state: int = 10

class TrainCVRequest(BaseModel):
    model_name: ModelName
    kfold_config: KFoldConfig = KFoldConfig()
    param_grid: Dict[str, List[Any]] = Field(default_factory=dict)
    standardize: bool = True

class TrainCVResultRow(BaseModel):
    prams: Dict[str, Any]
    mean_accuracy: float
    std_accuracy: float

class TrainCVResponse(BaseModel):
    best_params: Dict[str, Any]
    best_mean_accuracy: float
    best_std_accuracy: float
    results: List[TrainCVResultRow]

class PCAResponse(BaseModel):
    points: List[Dict[str, float]]