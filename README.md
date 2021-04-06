# cxr-annotation-inpainting

ML models used to detect COVID-19 from chest x-rays can use irrelevant features like annotations. This repository uses Pluralistic Image Inpainting to address this problem.

# Attribution

The repo modifies code from [https://github.com/lyndonzheng/Pluralistic-Inpainting](https://github.com/lyndonzheng/Pluralistic-Inpainting), a repo implementing  "Pluralistic Image Completion" by Chuanxia Zheng, Tat-Jen Cham and Jianfei Cai. This is test.py, train.py and the code in /dataloader, /evaluations, /model, /options, /util.

The modifications are:
* Adding a 80/20 train/test split to data_loader.py
* Updating base_options.py and data_loader.py to add a fourth mask option (detects and masks annotations on CXR)
* Implementing the fourth mask option in task.py (has_color(), custom_helper(), annotation_mask())

Other files:
* model_train_and_vis.ipynb is the original Colab used to run some experiments for "Using Grad-CAM to Improve Model Interpretability for COVID-19 and Viral Pneumonia Diagnosis from Chest X-ray Scans" by Irina Malyugina, Reha Matai, and Debanshi Misra. The training, testing, and validation code is written by Irina Malyugina. The Guided Grad-CAM code is from this repo: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam). The saliency maps code is written by Irina Malyugina using this toolkit: [https://github.com/MisaOgura/flashtorch](https://github.com/MisaOgura/flashtorch).
* resnet_train.py is an updated version of the code used to run final experiments for "Using Grad-CAM to Improve Model Interpretability for COVID-19 and Viral Pneumonia Diagnosis from Chest X-ray Scans" by Irina Malyugina, Reha Matai, and Debanshi Misra. It is written by Irina Malyugina.
* gradcam.py uses Grad-CAM, Guided Backprop, and Guided Grad-CAM to interpret the ResNet model's predictions. The code is from this repo: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).
* saliencymaps.py uses Backprop and Guided Backprop to interpret the ResNet model's predictions. It is written by Irina Malyugina using this toolkit: [https://github.com/MisaOgura/flashtorch](https://github.com/MisaOgura/flashtorch).

# Licensing

The code from [https://github.com/lyndonzheng/Pluralistic-Inpainting](https://github.com/lyndonzheng/Pluralistic-Inpainting) is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 

The code from [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) is licensed under an MIT license (see LICENSE-pytorch-grad-cam file).
