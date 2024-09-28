# DocLayNetPlus_mnbvc

this is sub_project of MNBVC, which is to aim to process DocLayNet dataset to MNBVC format.

**Steps:**

1. **Download and unzip:**
    * `wget -c https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip`
    * `unzip DocLayNet_core.zip`
    * `wget -c https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip`
    * `unzip DocLayNet_extra.zip`
2. **Update data_process.py:** provide local_path parameter of load_dataset method to the parent directory which 2 zip files have been extracted.
3. **Run:** `python data_process.py`
