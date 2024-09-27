# DocLayNetPlus_mnbvc
this is sub_project of MNBVC, which is to aim to process DocLayNet dataset to MNBVC format,

step#1: download and unzip:
wget -c https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip
unzip DocLayNet_core.zip

wget -c https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip
unzip DocLayNet_extra.zip

step#2: update data_process.py: provide local_path parameter of load_dataset method to the parent directory which 2 zip files have been extracted
step#3: python data_process.py
