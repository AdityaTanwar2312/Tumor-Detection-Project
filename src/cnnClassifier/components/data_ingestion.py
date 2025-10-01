import zipfile
import os
import gdown
from src.cnnClassifier import logger
from src.cnnClassifier.entity.config_entity import dataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: dataIngestionConfig):
        self.config = config

    def download_file(self) -> Path:
        if os.path.exists(self.config.local_data_file):
            logger.info(f"File already exists at {self.config.local_data_file}. Skipping download.")
        else:
            logger.info(f"Downloading file from {self.config.source_URL} to {self.config.local_data_file}")
            gdown.download(url=self.config.source_URL, output=self.config.local_data_file, quiet=False, fuzzy=True)
            logger.info(f"File downloaded successfully to {self.config.local_data_file}")
        return str(self.config.local_data_file)

    def extract_zip_file(self, zip_file_path: Path):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        logger.info("File path:", self.config.local_data_file)
        logger.info("Is zip file?", zipfile.is_zipfile(self.config.local_data_file))
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def initiate_data_ingestion(self):
        zip_file_path = self.download_file()
        self.extract_zip_file(zip_file_path)
        return self.config.unzip_dir