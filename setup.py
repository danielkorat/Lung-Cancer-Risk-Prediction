import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    requirements = fh.readlines()

    # [
    #     "Django >= 1.1.1",
    #     "caldav == 0.1.4",
    # ]

setuptools.setup(
    name="lungs",
    version="0.1.0",
    author="Daniel Korat",
    author_email="dkorat@gmail.com",
    description="3D Neural Network for Lung Cancer Risk Prediction on CT Volumes",
    long_description=long_description,
    url="https://github.com/danielkorat/Lung-Cancer-Risk-Prediction",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)