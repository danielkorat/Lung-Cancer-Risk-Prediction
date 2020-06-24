import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="lungs",
    version="0.1.2",
    author="Daniel Korat",
    author_email="dkorat@gmail.com",
    description="3D Neural Network for Lung Cancer Risk Prediction on CT Volumes",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/danielkorat/Lung-Cancer-Risk-Prediction",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)