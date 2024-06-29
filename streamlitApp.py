import os
import re
import json
import zipfile
import datetime
import numpy as np
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
import firebase_admin
from firebase_admin import credentials, auth, db

from DataAugmentationApp.ImageDataGeneration import ImageDataGeneratorComponent
from DataAugmentationApp.utils import create_directory, zipdir, clear_the_directories
from DataAugmentationApp.logger import logger

def dashboard(input_images_path, output_images_path, image_data_generator_object):
    
    # tabs for each way of augmentation techniques
    Home_tab, WorkFlow = st.tabs(["Home", "WorkFlow"])
    
    with Home_tab:
        st.subheader("Welcome to the Data Augmentation Application")
        st.markdown("This application is made to perform data augmentation with an easy to use interface. You are provided with various traditional data augmentation techniques like rotation, crop, zooming, etc.")
        st.markdown("The following :red[\"WorkFlow\"] tab will have the functionality of data augmentation. You can select the augmentation types and parameters to generate the augmented images.")
        st.markdown("""
            ### Perform Data Augmentation for Your Images

            - Go to the :red[**Workflow**] tab.
            - Upload the images :file_folder:.
            - Select the augmentation types below with appropriate parameters :gear:.
            - Click on the :red[**AUGMENTATE**] button.
            - Download the augmented images using the :red[**Download Images**] button.

            ---

            Hey! Want to find out more about the implementation of this application? Feel free to visit our OpenSource repository :blue[**[DataAugmentationApp](https://github.com/Govardhan211103/DataAugmentationApp/)**].
        """)

    with WorkFlow:
        st.subheader('Upload the images')
        # stream lit uploader in the side bar, can take multiple files together 
        uploaded_file = st.file_uploader("Choose", accept_multiple_files = True, key="WorkFlow")

        st.markdown('<h3>Select the augmentation types below</h3>', unsafe_allow_html=True)

        # list of numbers for selection box parameters
        range_of_floats = list(np.round((np.arange(0.1, 1.0, 0.1)), decimals = 1))

        default_brightness_range = (0.5, 1.5)

        # form for the input parameters
        with st.form(key='columns_in_form'):
            # two columns for easy access
            column1, column2 = st.columns(2)
            with column1:
                rotation_range = st.selectbox("**Rotation Range**", [None] + list(range(10, 180, 10)))
                shear_range = st.selectbox("**Shear Range**", [None] + list(range(10, 180, 10)))
                zoom_range = st.selectbox("**Zoom Range**", [None] + range_of_floats)
                width_shift_range = st.selectbox("**Crop Width Shift Range**", [None] + range_of_floats)
                 
            with column2:
                horizontal_flip = st.selectbox("**Horizontal Flip**", [None, True, False])
                vertical_flip = st.selectbox("**Vertical Flip**", [None, True, False])
                channel_shift_range = st.selectbox("**Channel Shift Range**", [None] + list(range(10, 100, 10)))
                height_shift_range = st.selectbox("**Crop Height Shift Range**", [None] + range_of_floats)
            brightness_range = st.slider('**Brightness Range**', 0.0, 2.0, (0.5, 1.5), step=0.1)
            no_of_images = st.number_input('**No of images for each technique**', min_value = 1, max_value = 10, value = 1)
               
            
            submitted = st.form_submit_button("**:red[AUGMENTATE]**", on_click= form_submission())


        # get the values of the form into parameters dictionary
        parameters_dict = {
            'rotator' : rotation_range,
            'shearer' : shear_range,
            'zoomer' : zoom_range, 
            'flipper_horizontal' : None if horizontal_flip is None or horizontal_flip is False else horizontal_flip, 
            'flipper_vertical' : None if vertical_flip is None or vertical_flip is False else vertical_flip,
            'cropper' : (width_shift_range, height_shift_range) if width_shift_range is not None and height_shift_range is not None else None,
            'brightness' : brightness_range if brightness_range != default_brightness_range else None,
            'channel_shift' : channel_shift_range if channel_shift_range is not None else None,
            'no_of_images' : no_of_images
        }
        logger.info("aquired the augmentation parameters.")

        if uploaded_file is not None:
            # save the uploaded file to the input_images_path directory
            for index, uploaded_image in enumerate(uploaded_file):
                image = uploaded_image.read()  # Read the image 

                input_file_path = os.path.join(input_images_path, f"{uploaded_image.name}.jpeg") 
                with open(input_file_path, "wb") as f:
                    f.write(uploaded_image.getvalue())
            logger.info("Saved the uploaded images to the input_images_path directory.")

        # call the get_augmentator_objects function to get the augmentator objects
        final_augmentators_dict, number_of_images = image_data_generator_object.get_augmentator_objects(parameters_dict)
        # perform the augmentation on the images in the output_images_path directory using image_augmentation function
        image_data_generator_object.image_augmentation(input_images_path, output_images_path, final_augmentators_dict, number_of_images)
        
        # call the download_files function
        download_images(input_images_path, output_images_path)


    # with OpenCV_tab:
    #     st.subheader('Upload the images')
    #     # stream lit uploader in the side bar, can take multiple files together 
    #     uploaded_file = st.file_uploader("Choose", accept_multiple_files = True, key = "OpenCV")
    #     st.markdown('<h3>Select the augmentation types below using OpenCV</h3>', unsafe_allow_html=True)
    # with TensorFlowLayers_tab:
    #     st.subheader('Upload the images')
    #     # stream lit uploader in the side bar, can take multiple files together 
    #     uploaded_file = st.file_uploader("Choose", accept_multiple_files = True, key = "TensorFlow")
    #     st.markdown('<h3>Select the augmentation types below using TensorFlow Layers</h3>', unsafe_allow_html=True)


def download_images(input_images_path, output_images_path):
    """
    This function creates a download button to download all the images in the output_images directory.
    """
    if len(os.listdir(output_images_path)) != 0:
        st.subheader("Download the images.")

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
    st.markdown('')
    st.markdown("**Download the augmented images Here!!**")
    download_button("**Download Images**", zip_file_path)


def form_submission():
    st.markdown("---")
    st.markdown("**Click the AUGMENTATE button to start the augmentation process.**")



def insert_user(users_reference, email, username, fullname, password):
    """
    Inserts Users into the DB
    :param users_reference:  #ref.child('usernames')
    :param email:
    :param username:
    :param fullname:
    :param password:
    :return:
    """
    date_joined = str(datetime.datetime.now())

    users_reference.update({
        username : {
            'name': fullname,
            'password': password,
            'email' : email,
            'date_joined': date_joined
        }
    })


def fetch_users():
    """
    Fetch Users
    :return Dictionary of Users:
    """
    reference = db.reference('/')
    return reference

def get_user_emails():
    """
    Fetch User Emails
    :return List of user emails:
    """
    ref = fetch_users()
    credentials = ref.get()
    user_data = credentials["usernames"]
    emails = [user_data['email'] for user_data in user_data.values()]
    
    return emails

def get_usernames():
    """
    Fetch Usernames
    :return List of user usernames:
    """
    ref = fetch_users()
    credentials = ref.get()
    user_data = credentials["usernames"]
    usernames = [user_data['name'] for user_data in user_data.values()]

    return usernames

def validate_email(email):
    """
    Check Email Validity
    :param email:
    :return True if email is valid else False:
    """
    pattern = "^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$"  #tesQQ12@gmail.com

    if re.match(pattern, email):
        return True
    return False

def signup_from_submission():
    pass
    # st.success("Your account has been created successfully. Please login to continue.")

def sign_up():
    with st.form(key='signup', clear_on_submit=True):
        # st.subheader(':green[Sign Up]')
        email = st.text_input(':blue[Email]', placeholder='Enter Your Email')
        fullname = st.text_input(':blue[Fullname]', placeholder='Enter Your Full Name')
        username = st.text_input(':blue[Username]', placeholder='Enter Your Username')
        password1 = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')
        password2 = st.text_input(':blue[Confirm Password]', placeholder='Confirm Your Password', type='password')
        balloons = None
        if email:
            if validate_email(email):
                if email not in get_user_emails():
                    if username not in get_usernames():
                        if len(username) >= 2:
                            if len(password1) >= 6:
                                if password1 == password2:
                                    # Add User to DB
                                    hashed_password = Hasher([password2]).generate()
                                    reference = fetch_users()
                                    insert_user(reference.child('usernames'), email, username, fullname, hashed_password[0])
                                    st.write(':green[Account created successfully!! PLEASE LOGIN TO CONTINUE]')
                                    st.balloons()
                                else:
                                    st.warning('Passwords Do Not Match')
                            else:
                                st.warning('Password is too Short')
                        else:
                            st.warning('Username Too short')
                    else:
                        st.warning('Username Already Exists')
                else:
                    st.warning('Email Already exists!!')
            else:
                st.warning('Invalid Email')

        st.form_submit_button(":red[\"SIGNUP\"]", on_click = signup_from_submission)


def main():
    logger.info("The Streamlit app has been initialised")
    st.set_page_config(page_title='Streamlit', initial_sidebar_state='collapsed')
    st.title("DataAug")
    st.markdown("Augment your images with ease!")
    st.markdown("""
        <style>
                .block-container {
                    padding-bottom: 0.5rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

    try:
        ref = fetch_users()
        credentials = ref.get()

        print(credentials)

        Authenticator = stauth.Authenticate(credentials, 'Streamlit124', 'abcdeff', cookie_expiry_days=1)

        # Button for navigation 
        # Initialize session state if not already set
        if 'page' not in st.session_state:
            st.session_state.page = 'Login'
        if 'clear_sidebar' not in st.session_state:
            st.session_state.clear_sidebar = False

        # Function to clear sidebar content
        def clear_sidebar():
            for key in st.session_state.keys():
                if key.startswith('sidebar_'):
                    del st.session_state[key]


        if st.session_state.page == 'Login':
            st.session_state.clear_sidebar = True  # Set flag to clear sidebar

            if st.session_state.clear_sidebar:
                clear_sidebar()
                st.session_state.clear_sidebar = False  # Reset the flag

            with st.sidebar:
                st.subheader("Welcome to DataAug")
                if st.button("Sign Up", key='signup'):
                    st.session_state.page = 'Sign Up'
                    st.experimental_rerun()

            name, authentication_status, username = Authenticator.login('main')

            info, info1 = st.columns(2)

            if authentication_status:
                st.sidebar.markdown("""---""")
                st.sidebar.title(f'Hey :red[{name}]!!')
                st.sidebar.markdown("Please click on logout to end your session")

                # define the paths for the input and output directories
                input_images_path = Path(os.path.join(os.getcwd(), 'data/input_images'))
                output_images_path = Path(os.path.join(os.getcwd(), 'data/output_images'))

                clear_the_directories(input_images_path, output_images_path)

                # create the input and output directories for the images 
                create_directory([input_images_path, output_images_path])

                # create the ImageDataGeneratorComponent object to access augmentation methods
                image_data_generator_object = ImageDataGeneratorComponent()

                dashboard(input_images_path, output_images_path, image_data_generator_object)

                # logout button using streamlit_authenticator
                Authenticator.logout(':red[Log Out]', 'sidebar')

            elif authentication_status == False:
                with info:
                    st.error('Incorrect Password or username')
            else:
                with info:
                    st.warning('Please feed in your credentials')

        elif st.session_state.page == 'Sign Up':
            with st.sidebar:
                st.subheader("Sign Up Page")
                if st.button("Go to Login"):
                    st.session_state.page = 'Login'
                    st.experimental_rerun()

            sign_up()


    except Exception as e:
        print("the error ", e)
        st.success('Refresh Page')



if __name__ == '__main__':
    try:
        # if not firebase_admin._apps:
        firebase_config = st.secrets["firebase"]
        private_key = firebase_config["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate({
            "type": firebase_config["type"],
            "project_id": firebase_config["project_id"],
            "private_key_id": firebase_config["private_key_id"],
            "private_key": private_key,
            "client_email": firebase_config["client_email"],
            "client_id": firebase_config["client_id"],
            "auth_uri": firebase_config["auth_uri"],
            "token_uri": firebase_config["token_uri"],
            "auth_provider_x509_cert_url": firebase_config["auth_provider_x509_cert_url"],
            "client_x509_cert_url": firebase_config["client_x509_cert_url"]
        })
        firebase_admin.initialize_app(cred, {
            'databaseURL': st.secrets["databaseURL"]["url"]
        })
        # app = firebase_admin.get_app()
    except ValueError as e:
        print("Initialization error : ", e)
    
    main()

