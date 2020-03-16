Documentation for CDAS file package

Description: This file describes the components of this CDAS data delivery package.


Main Directory
File List:
1. Delivery File Readme: README.txt


Directory: Standard 15K
Description: This zip file contains 3 data sets (in CSV format) and 3 corresponding data dictionaries.
It also contains an image ID list containing PIDs for CT image selected population.
The data sets are as follows:
- prsn: one record per person in NLST, with a flag variable (CT_SELECTED) to identify the sub-population with images.
- ct_ab: one record per abnormality seen on CT screening exams. Includes nodule features.
- image: one record per CT image series.
Includes IDs for linking to CT image files, as well as data extracted from images' DICOM headers.
File List:
1. CSV Dataset: nlst15k_delivery_20180720.zip


