# Impact of Disease Prevalence and Data Distribution on Fairness Guarantees in Equitable Deep Learning

## Dataset
We used two datasets, **[FairVision](https://ophai.hms.harvard.edu/datasets/harvard-fairvision30k)** and **[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)**, for the empirical validation of the theoretical findings.



## Experiments

### AMD Detection
To run the experiments with the baseline models (e.g., ViT) on the task of AMD detection, execute:
```bash
python train_amd_vit_fea.py
```
Or using the EfficientNet model:
```
python train_amd_efficientnet_fea.py
```

### DR Detection
To run the experiments with the baseline models (e.g., ViT) on the task of DR detection, execute:
```bash
python train_dr_vit_fea.py
```
Or using the EfficientNet model:
```
python train_dr_efficientnet_fea.py
```

### Glaucoma Detection
To run the experiments with the baseline models (e.g., ViT) on the task of Glaucoma detection, execute:
```bash
python train_glaucoma_vit_fea.py
```
Or using the EfficientNet model:
```
python train_glaucoma_efficientnet_fea.py
```

### Pleural Effusion Detection
To run the experiments with the baseline models (e.g., ViT) on the task of AMD detection, execute:
```bash
python train_chexpert_vit_fea.py
```
Or using the EfficientNet model:
```
python train_chexpert_efficientnet_fea.py
```
