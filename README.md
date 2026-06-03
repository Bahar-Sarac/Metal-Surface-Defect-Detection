In this project, I am training a model for metal surface defect detection, leveraging the open-source NEU-DET and GC10-DET datasets. The combined and customized dataset, named "https://app.roboflow.com/bahs-work-space/metal-defect-detection-neu-det-gc10-det/3", was curated and organized using Roboflow.
The dataset contains a total of 8 distinct defect classes. The detailed distribution of the dataset across training, validation, and test splits is provided in the table below:
| Class ID | Defect Name | Train | Valid | Test | TOTAL |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | crazing | 549 | 57 | 82 | **688** |
| **1** | fold | 157 | 23 | 23 | **203** |
| **2** | hole | 479 | 53 | 55 | **587** |
| **3** | inclusion | 1069 | 127 | 158 | **1354** |
| **4** | patches | 1507 | 136 | 155 | **1798** |
| **5** | pitted_surface | 396 | 60 | 59 | **515** |
| **6** | rolled_in_scale | 521 | 54 | 53 | **628** |
| **7** | scratches | 1139 | 159 | 134 | **1432** |
| **-** | **TOTAL** | **5817** | **669** | **719** | **7205** |
