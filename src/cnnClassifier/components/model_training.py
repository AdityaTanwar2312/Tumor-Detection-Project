import os
from urllib.request import Request
from zipfile import ZipFile
import tensorflow as tf
import time
from src.cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train_valid_generator(self):
        datagen_kwargs = dict(rescale=1./255, validation_split=0.20)
        dataflow_kwargs = dict(target_size=self.config.param_image_size[:2],
                               batch_size=self.config.param_batch_size,
                               interpolation="bilinear")

        if self.config.param_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.2,
                **datagen_kwargs
            )
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

        self.train_generator = train_datagen.flow_from_directory(
            directory=self.config.training_data.as_posix(),
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            directory=self.config.training_data.as_posix(),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
    
    def load_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        # Recompile with a new optimizer instance
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.param_learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    @staticmethod
    def save_model(model, path: Path):
        model.save(path)
    
    def train(self):
        self.step_per_epoch = self.train_generator.samples // self.config.param_batch_size
        self.validation_steps = self.valid_generator.samples // self.config.param_batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.param_epochs,
            steps_per_epoch=self.step_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
        )
        self.save_model(model=self.model, path=self.config.trained_model_path)