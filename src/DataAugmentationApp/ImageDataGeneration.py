
import os
import zipfile
import numpy as np
from pathlib import Path
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from DataAugmentationApp.utils import zipdir, create_directory
from DataAugmentationApp.logger import logger

class ImageDataGeneratorComponent():
    def __init__(self):
        pass

    def get_augmentator_objects(self, parameters_dict):
        """
        This function returns a dictionary of ImageDataGenerators based on the 
        augmentations specified in the parameters_dict.

        The function will return a dictionary with the ImageDataGenerator objects

        The function will return only the ImageDataGenerators that have a non-None
        value in the parameters_dict. If a particular augmentation is not specified
        in the parameters_dict, that ImageDataGenerator will not be included in the
        dictionary.

        Args: 
            parameters_dict (dict): dictionary containing the augmentation parameters
        returns:
            augmentators_dict (dict): dictionary containing the ImageDataGenerator objects
        """

        rotator = ImageDataGenerator(rotation_range = parameters_dict['rotator'], fill_mode = 'nearest') if parameters_dict['rotator'] is not None else None
        cropper = ImageDataGenerator(width_shift_range = parameters_dict['cropper'][0], height_shift_range = parameters_dict['cropper'][1], fill_mode = 'nearest') if parameters_dict['cropper'] is not None else None
        shearer = ImageDataGenerator(shear_range = parameters_dict['shearer'], fill_mode = 'nearest') if parameters_dict['shearer'] is not None else None
        zoomer = ImageDataGenerator(zoom_range = parameters_dict['zoomer'], fill_mode = 'nearest') if parameters_dict['zoomer'] is not None else None
        flipper_horizontal = ImageDataGenerator(horizontal_flip = parameters_dict['flipper_horizontal']) if parameters_dict['flipper_horizontal'] is not None else None
        flipper_vertical = ImageDataGenerator(vertical_flip = parameters_dict['flipper_vertical']) if parameters_dict['flipper_vertical'] is not None else None
        brightness = ImageDataGenerator(brightness_range = parameters_dict['brightness']) if parameters_dict['brightness'] is not None else None
        channel_shift = ImageDataGenerator(channel_shift_range = parameters_dict['channel_shift']) if parameters_dict['channel_shift'] is not None else None
        no_of_images = parameters_dict['no_of_images']

        logger.info("created ImageDataGenerator objects")

        augmentators_dict = {
            'rotator' : rotator ,
            'cropper' : cropper ,
            'shearer' : shearer ,
            'zoomer' : zoomer ,
            'flipper_horizontal' : flipper_horizontal ,
            'flipper_vertical' : flipper_vertical ,
            'brightness' : brightness ,
            'channel_shift' : channel_shift
        }
        final_augmentators_dict = {key: value for key, value in augmentators_dict.items() if value is not None}

        logger.info("returning ImageDataGenerator objects")
        return (final_augmentators_dict, no_of_images)

    
    def image_augmentation(self, input_images_path, output_images_path, final_augmentators_dict, number_of_images):
        """
        This function loads each image from the input_images_path directory,
        applies the final_augmentators_dict on it and saves the augmented images
        to the output_images_path directory.

        Args:
            input_images_path (str): path to the directory containing the input images
            output_images_path (str): path to the directory where the augmented images will be saved
            final_augmentators_dict (dict): dictionary containing the ImageDataGenerator objects
        returns:
            None
        """
        images_paths = os.listdir(input_images_path)
        # [img1.png, img2.png]
        if len(images_paths) != 0:
            for index, path in enumerate(images_paths):
                image_path = os.path.join(input_images_path, path)
                # Load a sample input image
                img = load_img(image_path)
                logger.info(f"Image {image_path} is loaded")
                
                # Convert the image to a numpy array
                array = img_to_array(img)
                logger.info(f"Image array shape : {array.shape}")

                array = array.reshape((1,) + array.shape)

                # save the augmented images to the output_directory file
                for i, (augmentator_key, augmentator) in enumerate(final_augmentators_dict.items()):
                    for j in range(number_of_images):
                        # get the augmentated image
                        augmented_img = augmentator.flow(array, batch_size=1)[0]
                        # printing the augmented image shape in terminal
                        print(augmented_img.shape)

                        # Convert the augmented image back to a PIL Image
                        augmented_img = augmented_img.reshape(augmented_img.shape[1:])
                        augmented_img = tf.keras.preprocessing.image.array_to_img(augmented_img)

                        # output file path
                        image_file_name = f"{path.split('.')[0]}_{augmentator_key}{j}.jpeg" 
                        output_file_path = os.path.join(output_images_path, image_file_name)
                        # Save the image to the specified path
                        augmented_img.save(output_file_path)

                        st.write(f":green[Augmented {image_file_name}]")
                        logger.info(f"Augmented {image_file_name}")

                logger.info(f"Image {image_path} has been augmented")
        else:
            logger.info(f"No images found in {input_images_path}")


if __name__ == '__main__':
    logger.info("The Image Generator Component has been initialised")