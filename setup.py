from setuptools import setup, find_packages

with open("README.md", "r", encoding = 'utf-8') as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "DataAugmentationApp"
AUTHOR_USER_NAME = "Govardhan211103"
SRC_REPO = "DataAugmentationApp"
AUTHOR_EMAIL = "vgovardhanvarma.vh@gmail.com"

setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "Data Augmentation application",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    # url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    # project_urls = {
    #     "Bug Tracker" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    # },
    package_dir = {"": "src"},
    packages = find_packages(where = "src")
)