## Model Zoo
If you only want to use our trained checkpoints for inference or fine-tuning, here is the collection of models.

Models with 3 experts are standard models that provide a good trade-off between computation/model size and accuracy. Note that models with 2 experts sometimes have even lower computational cost than baseline models. However, we will also release some models that achieve higher accuracy such as models with 6 experts, which can be used as teacher models to distill other models. Some models are trained in an old config format so that config may mismatch. If you cannot load the checkpoint, please let us know.

### Imbalanced CIFAR 100/CIFAR-LT 100 (100 epochs)
1. CE and Decouple: baseline results for cross-entropy and decouple (cRT/tau-norm/LWS)
2. RIDE: ResNet32 backbone, without distillation, with EA
3. RIDE + Distill: ResNet32 backbone, with distillation, with EA
4. Teacher Model: ResNet32 backbone, 6 experts, without EA. Working as the teacher model when optimizing RIDE with knowledge distillation.

<!--
Directory name:
1. cifar_standard_055148
2. cifar_standard_distill_003240
3. cifar_large_053612
4. cifar_teacher_015420
-->

| Model          | #Experts | Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | ---------------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| CE                 | - | 39.1         | 66.1         | 37.3           | 10.6        | -
| Decouple           | - | 43.3         | 64.0         | 44.8           | 18.1        | -
| **RIDE**           | 3 | 48.6         | 67.0         | 49.9           | 25.7        | [Link](https://drive.google.com/file/d/1uE8I_2JcslWGPu4O0nAFEIk7iR_Sw5lS/view?usp=sharing)
| **RIDE + Distill** | 3 | 49.0         | 67.6         | 50.9           | 25.2        | [Link](https://drive.google.com/file/d/1W-EICEpAavKzlnayiFPvb5cDyGCBl34l/view?usp=sharing)
| **RIDE + Distill** | 4 | 49.4         | 67.7         | 51.3           | 25.7        | [Link](https://drive.google.com/file/d/11kyxcYIh3bXk3mn3Y8EENHcsx-Ld9PXH/view?usp=sharing)
| *Teacher Model*    | 6 | 50.2         | 69.3         | 52.1           | 25.8        | [Link](https://drive.google.com/file/d/1kq8SaoHUujqIOplsKUNRpKM7UQR0qg-k/view?usp=sharing)

### ImageNet-LT (100 epochs)
1. CE and Decouple: baseline results for cross-entropy and decouple (cRT/tau-norm/LWS)
2. RIDE: ResNeXt50 backbone, 3 experts, without distillation, with EA
2. RIDE + Distill: ResNeXt50 backbone, with distillation, with EA
3. Teacher Model: ResNeXt50 backbone, 6 experts, without EA. Working as the teacher model when optimizing RIDE with knowledge distillation.

<!--
Directory name:
1. imagenet_lt_standard_051430
2. imagenet_lt_larger_distill_133441
3. imagenet_lt_teacher_084702
-->

| Model          | #Experts | Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | ---------------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| CE                 | - | 44.4              | 65.9          | 37.5            | 7.7          | -
| Decouple           | - | 49.9              | 60.2          | 47.2            | 30.3         | -
| **RIDE**           | 3 |  55.7             | 67.0          | 52.2            | 36.0        | [Link](https://drive.google.com/file/d/1d4PHfWZ_rfTRDIJG5sogK1cO0BRoi9d9/view?usp=sharing)
| **RIDE + Distill** | 4 |  56.8             | 68.3          | 53.5            | 35.9        | [Link](https://drive.google.com/file/d/1G3aT7YzEixb0mSQBpZpuUfTT3b9YsSbz/view?usp=sharing)
| *Teacher Model*    | 6 |  57.5             | 68.9          | 54.3            | 36.5        | [Link](https://drive.google.com/file/d/1hJyMgbv0JSisXCiHpC1xcHhbGXJP8K8a/view?usp=sharing)

### iNaturalist (100 epochs)
1. CE and Decouple: baseline results for cross-entropy and Decouple (cRT/tau-norm/LWS)
2. RIDE: ResNet50 backbone, without distillation, with EA
3. RIDE + Distill: ResNet50 backbone, with distillation, with EA (in FP16)
4. Teacher Model: ResNet50 backbone, 6 experts, without EA. Working as the teacher model when optimizing RIDE with knowledge distillation.

<!--
Directory name:
1. iNaturalist_standard_191630
2. iNaturalist_large_182137
3. iNaturalist_teacher_104314
-->

| Model          | #Experts |  Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | ---------------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| CE                 | - | 61.7              | 72.2          | 63.0            | 57.2         | -
| Decouple           | - | 65.9              | 65.0          | 66.3            | 65.5         | -
| **RIDE**           | 3 | 71.2              | 70.2          | 71.2            | 71.6        | [Link](https://drive.google.com/file/d/1KVrKrQXsuzeeb2oFzjloEf2XrvIfb42u/view?usp=sharing)
| **RIDE + Distill** | 4 | 72.6              | 70.9          | 72.5            | 73.1        | [Link](https://drive.google.com/file/d/1PdfWVQlsTjPFDr7bTFeUUskh2RA6Mb_r/view?usp=sharing)
| *Teacher Model*    | 6 | 72.9              | 71.1          | 72.9            | 73.3        | [Link](https://drive.google.com/file/d/1DtLlx3be7WCmtVzoGBSGCiImQDJNxHGJ/view?usp=sharing)

#### iNaturalist (Longer Training)
1. RIDE + Distill: ResNet 50 backbone, 4 experts, with EA, 200 epochs (distilled from 6 experts, 200 epochs).
2. RIDE: ResNet 50 backbone with 6 experts, without EA, 300 epochs.

| Model          | #Experts | Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | -------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| **RIDE + Distill** | 4 |      73.2           | 70.5          | 73.7            | 73.3         | [Link](https://drive.google.com/drive/folders/1Kz-SwP6vRx7ktZhYWLmJLG6uprkj38vp?usp=sharing)
| **RIDE**           | 6 |      74.6           | 71.0          | 75.7            | 74.3         | [Link](https://drive.google.com/drive/folders/1fyPJdgsLLTA7JE6uzUZmPanh1e1I8rKy?usp=sharing)

After downloading the checkpoints, you could run evaluation by following the instructions in the test section.
