from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_training import ModelTrainer
from src.cnnClassifier import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        model_trainer = ModelTrainer(config=training_config)
        model_trainer.load_base_model()
        model_trainer.train_valid_generator()
        model_trainer.train()

if __name__ == "__main__":
    try:
        logger.info("***********************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e