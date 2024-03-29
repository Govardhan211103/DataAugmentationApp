import os
from pathlib import Path
from DataAugmentationApp.logger import logger

# Function to zip a directory
def zipdir(path, zipf):
    """
    This function zips a directory and all its contents recursively.

    Args:
        path (str): path to the directory to be zipped
        zipf (ZipFile): ZipFile object to write the zipped data to
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), path))
    logger.info(f"{zipf} has been zipped")

def create_directory(path_to_directories: list, verbose = True):
    """
    Create a list of directories.
    Args:
        path_to_directories(list): list of paths of directories to create
        ignore_log(bool, optional): ignore if multiple dirs to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(Path(path), exist_ok=True)
        if verbose:
            logger.info(f"directory: {path} created successfully")
