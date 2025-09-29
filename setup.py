import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version__ = "0.0.0"

REPO_NAME = "Tumor-Detection-Project"
AUTHOR_USER_NAME = "AdityaTanwar2312"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "KSDJKGNV@SJESJ.hmakij"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small package for CNN based image classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)