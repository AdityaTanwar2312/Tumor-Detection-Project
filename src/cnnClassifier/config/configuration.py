import os
from pathlib import Path
from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity.config_entity import (dataIngestionConfig, 
                                                    PrepareBaseModelConfig,
                                                    TrainingConfig)


class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([str(self.config.artifact_root)])

    def get_data_ingestion_config(self) -> dataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = dataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=str(Path(config.local_data_file)),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_path = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,
            params_include_top = self.params.INCLUDE_TOP,
            params_classes = self.params.CLASSES,
            params_weights = self.params.WEIGHTS
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training_config = self.config.training

        root_dir = Path(training_config.root_dir)
        create_directories([root_dir])

        training_data = Path(self.config.data_ingestion.unzip_dir)
        updated_base_model_path = Path(self.config.prepare_base_model.updated_base_model_path)

        param_epochs = self.params.EPOCHS
        param_batch_size = self.params.BATCH_SIZE
        param_image_size = self.params.IMAGE_SIZE
        param_is_augmentation = self.params.AUGMENTATION

        training_config = TrainingConfig(
            root_dir=root_dir,
            trained_model_path=Path(training_config.trained_model_path),
            training_data=training_data,
            updated_base_model_path=updated_base_model_path,
            param_epochs=param_epochs,
            param_batch_size=param_batch_size,
            param_image_size=param_image_size,
            param_is_augmentation=param_is_augmentation,
            param_learning_rate=self.params.LEARNING_RATE
        )

        return training_config