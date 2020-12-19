## Model Zoo
If you only want to use our trained checkpoints for inference or fine-tuning, here is the collection of models.

Models with 3 experts are standard models that provide a good balance between computation/model size and accuracy and should be used to compare with other models fairly. Note that models with 2 experts sometimes have even lower computational cost than baseline models. However, we will also release some models that achieve higher accuracy such as models with 6 experts, which can be used as teacher models to distill other models. Some models are trained in an old config format so that config may mismatch. If you cannot load the checkpoint, please tell us.

### Imbalanced CIFAR 100/CIFAR-LT 100
1. CE and Decouple: baseline results for cross-entropy and current SOTA decouple (cRT/tau-norm/LWS)
2. RIDE: ResNet32 backbone, without distillation, with EA
3. RIDE + Distill: ResNet32 backbone, with distillation, with EA
<!-- 3. RIDE's larger Model with better performance: ResNet32 backbone, 4 experts, with distillation, with EA -->
4. Teacher Model: ResNet32 backbone, 6 experts, without EA

<!--
Directory name:
1. cifar_standard_055148
2. cifar_standard_distill_003240
3. cifar_large_053612
4. cifar_teacher_015420
-->

| Model          | #Experts | Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | ---------------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| CE             | - | 39.1         | 66.1         | 37.3           | 10.6        | -
| Decouple       | - | 43.3         | 64.0         | 44.8           | 18.1        | -
| RIDE           | 3 | 48.6         | 67.0         | 49.9           | 25.7        | Link
| RIDE + Distill | 3 | 49.0         | 67.6         | 50.9           | 25.2        | Link
| RIDE + Distill | 4 | 49.4         | 67.7         | 51.3           | 25.7        | Link
| Teacher Model  | 6 | 50.2         | 69.3         | 52.1           | 25.8        | Link

### ImageNet_LT
1. CE and Decouple: baseline results for cross-entropy and current SOTA decouple (cRT/tau-norm/LWS)
2. RIDE: ResNeXt50 backbone, 3 experts, without distillation, with EA
2. RIDE + Distill: ResNeXt50 backbone, with distillation, with EA
3. Teacher Model: ResNeXt50 backbone, 6 experts, without EA

<!--
Directory name:
1. imagenet_lt_standard_051430
2. imagenet_lt_larger_distill_133441
3. imagenet_lt_teacher_084702
-->

| Model          | #Experts | Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | ---------------- | ---------------- | ------------- | --------------- | ------------ | -------- |'
| CE             | - | 44.4              | 65.9          | 37.5            | 7.7          | -
| Decouple       | - | 49.9              | 60.2          | 47.2            | 30.3         | -
| RIDE           | 3 |  55.7             | 66.85         | 52.32           | 36.51        | Link
| RIDE + Distill | 4 |  56.8             | 68.28         | 53.52           | 35.93        | Link
| Teacher Model  | 6 |  57.5             | 68.85         | 54.35           | 36.53        | Link

### iNaturalist
1. CE, Decouple, BBN: baseline results for cross-entropy, Decouple (cRT/tau-norm/LWS) and current SOTA BBN
2. RIDE: ResNet50 backbone, without distillation, with EA
3. RIDE + Distill: ResNet50 backbone, with distillation, with EA (in FP16)
4. Teacher Model: ResNet50 backbone, 6 experts, without EA

<!--
Directory name:
1. iNaturalist_standard_191630
2. iNaturalist_large_182137
3. iNaturalist_teacher_104314
-->

| Model          | #Experts |  Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| CE             | - | 61.7              | 72.2          | 63.0            | 57.2         | -
| Decouple       | - | 65.9              | 65.0          | 66.3            | 65.5         | -
| BBN            | - | 66.3              | 49.4          | 70.8            | 65.3         | -
| RIDE           | 3 | 71.2              | 70.19         | 71.20           | 71.56        | Link
| RIDE + Distill | 4 | 72.6              | 70.93         | 72.39           | 73.11        | Link
| Teacher Model  | 6 | 72.8              | 71.02         | 72.87           | 73.27        | Link


After downloading the checkpoints, you could run evaluation by following the instructions in the test section below.