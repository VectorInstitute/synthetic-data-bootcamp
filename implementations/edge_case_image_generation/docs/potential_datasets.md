### **Transportation / Inspection**


|                           |                          |                                                                             |                                                                                                                                                             |
| ------------------------- | ------------------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**               | **Size**                 | **Rare-condition angle**                                                    | **Link**                                                                                                                                                    |
| RailSem19                 | ~8k images               | Rare track obstructions, signals in fog/night                               | [github.com/ybarancan/rail-dataset-tools](https://github.com/ybarancan/rail-dataset-tools) · [https://wilddash.cc/railsem19](https://wilddash.cc/railsem19) |
| NEU Surface Defect        | 1,800 images (6 classes) | 300 images per defect class — very amenable to augmentation                 |                                                                                                                                                             |
| DAGM 2007                 | ~6k images               | Weakly labeled industrial surface defects                                   | [conferences.mpi-inf.mpg.de/dagm/2007](http://conferences.mpi-inf.mpg.de/dagm/2007)                                                                         |
| ACDC (Adverse Conditions) | 4k images                | Specifically partitioned: fog, night, rain, snow — natural long-tail splits | [acdc.vision.ee.ethz.ch](http://acdc.vision.ee.ethz.ch)                                                                                                     |
| CARLA + nuScenes mini     | ~400 scenes              | Sensor-rich; useful for geometry-controlled ControlNet seeds                | [nuScenes mini](https://www.nuscenes.org/nuscenes#download)                                                                                                 |


### **Healthcare**


|                       |                                                            |                                            |                                                                                                                |
| --------------------- | ---------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **Dataset**           | **Size**                                                   | **Rare-condition angle**                   | **Link**                                                                                                       |
| VinDr-CXR             | 18k X-rays (but ~15 pathology classes are very imbalanced) | Aortic enlargement, ILD < 200 images each  | [physionet.org/content/vindr-cxr](https://physionet.org/content/vindr-cxr/1.0.0/)                              |
| SIIM-ACR Pneumothorax | 12k; positive rate ~3%                                     | Ideal "rare positive" scenario             | [kaggle.com/c/siim-acr-pneumothorax-segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) |
| Retinal OCT (Kermany) | 84k, but "drusen" and "CNV" << others                      | Class-imbalanced medical imaging benchmark | [data.mendeley.com/datasets/rscbjbr9sj/3](https://data.mendeley.com/datasets/rscbjbr9sj/3)                     |


### **Telecom / Infrastructure Inspection**


|                              |             |                                                                     |                                                                                                                                    |
| ---------------------------- | ----------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**                  | **Size**    | **Rare-condition angle**                                            | **Link**                                                                                                                           |
| COCO subset — infrastructure | Filterable  | Cell tower damage, cable damage, corrosion patches are rare classes | [cocodataset.org](http://cocodataset.org)                                                                                          |
| Wind Turbine Blade Defects   | ~600 images | Very small, perfect proof-of-rare-data scenario                     | [kaggle.com/datasets/ajifoster3/wind-turbine-blade-defects](https://www.kaggle.com/datasets/ajifoster3/wind-turbine-blade-defects) |


  
  
