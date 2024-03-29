import os
import zipfile
import numpy as np
from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from DataAugmentationApp.ImageDataGeneration import ImageDataGeneratorComponent
from DataAugmentationApp.utils import create_directory, zipdir
from DataAugmentationApp.logger import logger

def main(input_images_path, output_images_path, image_data_generator_object):
    # title
    st.header('DataAug')
    # description 
    st.markdown(":dart: The StreamLit app is made to perform data augmentation with an easy to use interface.")
    # tabs for each way of augmentation techniques
    ImageDataGenerator_tab, OpenCV_tab, TensorFlowLayers_tab = st.tabs(["ImageDataGenerator", "OpenCV", "TensorFlowLayers"])

    with ImageDataGenerator_tab:
        st.markdown('<h3>Select the augmentation types below using ImageDataGenerator</h3>', unsafe_allow_html=True)
        st.sidebar.title('Give the images')

        # list of numbers for selection box parameters
        range_of_floats = list(np.round((np.arange(0.1, 1.0, 0.1)), decimals = 1))

        # form for the input parameters
        with st.form(key='columns_in_form'):
            # two columns for easy access
            column1, column2 = st.columns(2)
            with column1:
                rotation_range = st.selectbox("Rotation Range", [None] + list(range(10, 180, 10)))
                shear_range = st.selectbox("Shear Range", [None] + list(range(10, 180, 10)))
                zoom_range = st.selectbox("Zoom Range", [None] + range_of_floats)
                width_shift_range = st.selectbox("Crop Width Shift Range", [None] + range_of_floats)
                
            with column2:
                horizontal_flip = st.selectbox("Horizontal Flip", [None, True, False])
                vertical_flip = st.selectbox("Vertical Flip", [None, True, False])
                brightness_range = st.selectbox("Brightness Range", [None] + range_of_floats)
                height_shift_range = st.selectbox("Crop Height Shift Range", [None] + range_of_floats)
            
            submitted = st.form_submit_button("Get the images", on_click= form_submission())


        # get the values of the form into parameters dictionary
        parameters_dict = {
            'rotator' : rotation_range,
            'shearer' : shear_range,
            'zoomer' : zoom_range, 
            'flipper_horizontal' : None if horizontal_flip is None or horizontal_flip is False else horizontal_flip, 
            'flipper_vertical' : None if vertical_flip is None or vertical_flip is False else vertical_flip,
            'cropper' : (width_shift_range, height_shift_range) if width_shift_range is not None and height_shift_range is not None else None,
            'brightness' : (0, brightness_range) if brightness_range is not None else None
        }
        logger.info("aquired the augmentation parameters.")
            
        # stream lit uploader in the side bar, can take multiple files together 
        uploaded_file = st.sidebar.file_uploader("Choose a file", accept_multiple_files = True)

        if uploaded_file is not None:
            # save the uploaded file to the input_images_path directory
            for index, uploaded_image in enumerate(uploaded_file):
                image = uploaded_image.read()  # Read the image 

                input_file_path = os.path.join(input_images_path, f"{uploaded_image.name}.jpeg") 
                with open(input_file_path, "wb") as f:
                    f.write(uploaded_image.getvalue())
            logger.info("Saved the uploaded images to the input_images_path directory.")

        # call the get_augmentator_objects function to get the augmentator objects
        final_augmentators_dict = image_data_generator_object.get_augmentator_objects(parameters_dict)
        # perform the augmentation on the images in the output_images_path directory using image_augmentation function
        image_data_generator_object.image_augmentation(input_images_path, output_images_path, final_augmentators_dict)
        
        # call the download_files function
        download_images(input_images_path, output_images_path)


    with OpenCV_tab:
        st.markdown('<h3>Select the augmentation types below using OpenCV</h3>', unsafe_allow_html=True)
    with TensorFlowLayers_tab:
        st.markdown('<h3>Select the augmentation types below using TensorFlow Layers</h3>', unsafe_allow_html=True)


def download_images(input_images_path, output_images_path):
    """
    This function creates a download button to download all the images in the output_images directory.
    """
    def download_button(label, file_path, button_text="Download"):
        """
        This function creates a download button to download the specified file.

        Args:
            label (str): label for the download button
            file_path (str): path to the file to be downloaded
            button_text (str): text for the download button

        Returns:
            None
        """
        with open(file_path, 'rb') as f:
            zip_bytes = f.read()
        download_button = st.download_button(label, zip_bytes, file_name=os.path.basename(file_path), mime="application/zip")
        if download_button:
            logger.info("================== download button clicked ====================")
    # Path to your directory of images
    directory_path = output_images_path

    # Zip the directory
    zip_file_path = 'images.zip'  # Temporary path for the zip file
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipdir(directory_path, zipf)

    # Display the download button
    download_button("Download Images", zip_file_path)


def form_submission():
    st.write("Data has been generated click the download button")

if __name__ == '__main__':
    logger.info("The Streamlit app has been initialised")
    # define the paths for the input and output directories
    input_images_path = Path(os.path.join(os.getcwd(), 'data/input_images'))
    output_images_path = Path(os.path.join(os.getcwd(), 'data/output_images'))

    # create the input and output directories for the images 
    create_directory([input_images_path, output_images_path])

    # create the ImageDataGeneratorComponent object to access augmentation methods
    image_data_generator_object = ImageDataGeneratorComponent()

    main(input_images_path, output_images_path, image_data_generator_object)
    

