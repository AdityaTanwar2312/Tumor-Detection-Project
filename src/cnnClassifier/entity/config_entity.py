from dataclasses import dataclass
from pathlib import Path

# Dataclass is just a decorator that automatically adds special methods to classes, 
# like __init__() and __repr__(), based on class attributes. 

# It simplifies the creation of classes that primarily store data.
@dataclass(frozen=True)
class dataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_path: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_classes: int
    params_weights: str

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    training_data: Path
    updated_base_model_path: Path
    param_learning_rate: float
    param_epochs: int
    param_batch_size: int
    param_image_size: list
    param_is_augmentation: bool