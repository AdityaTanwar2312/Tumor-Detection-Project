from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.cnnClassifier.pipeline.prepare_base_model_pipeline import PrepareBaseModelTrainingPipeline
from src.cnnClassifier.pipeline.model_training_pipeline import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:      
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"
try:
    logger.info('******************')
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training Stage"
try:
    logger.info("***********************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:    
    logger.exception(e)
    raise e