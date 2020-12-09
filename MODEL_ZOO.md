## Model Zoo
If you only want to use our trained checkpoints for inference or fine-tuning, here is the collection of models.

Models with 3 experts are standard models that provide a good balance between computation/model size and accuracy and should be used to compare with other models fairly. Note that models with 2 experts sometimes have even lower computation than standard model, so please don't use the models with 2 experts for comparisons. However, we will also release some models that achieve higher accuracy such as models with 6 experts, which can be used as teacher models to distill other models. Some models are trained in an old config format so that config may mismatch. If you cannot load the checkpoint, please tell us.

### Imbalanced CIFAR 100/CIFAR-LT 100
1. Standard Model: ResNet32 backbone, 3 experts, without distillation, with EA
2. Standard Model + Distill: ResNet32 backbone, 3 experts, with distillation, with EA
3. Larger Model with better performance: ResNet32 backbone, 4 experts, with distillation, with EA
4. Teacher Model: ResNet32 backbone, 6 experts, without EA

<!--
Directory name:
1. cifar_standard_055148
2. cifar_standard_distill_003240
3. cifar_large_053612
4. cifar_teacher_015420
-->

| Model          | Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| Standard Model | 48.6             | 66.97         | 49.94           | 25.73        | Link
| Standard Model + Distill | 49.0   | 67.6          | 50.89           | 25.23        | Link
| Larger Model   | 49.4             | 67.74         | 51.26           | 25.70        | Link
| Teacher Model  | 50.2             | 69.31         | 52.09           | 25.83        | Link

### ImageNet_LT
1. Standard Model: ResNeXt50 backbone, 3 experts, without distillation, with EA
3. Larger Model with better performance: ResNeXt50 backbone, 4 experts, with distillation, with EA
4. Teacher Model: ResNeXt50 backbone, 6 experts, without EA

<!--
Directory name:
1. imagenet_lt_standard_051430
2. imagenet_lt_larger_distill_133441
3. imagenet_lt_teacher_084702
-->

| Model          | Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| Standard Model | 55.7             | 66.85         | 52.32           | 36.51        | Link
| Larger Model   | 56.8             | 68.28         | 53.52           | 35.93        | Link
| Teacher Model  | 57.5             | 68.85         | 54.35           | 36.53        | Link

### iNaturalist
1. Standard Model: ResNet50 backbone, 3 experts, without distillation, with EA
2. Larger Model with better performance: ResNet50 backbone, 4 experts, with distillation, with EA (in FP16)
3. Teacher Model: ResNet50 backbone, 6 experts, without EA

<!--
Directory name:
1. iNaturalist_standard_191630
2. iNaturalist_large_182137
3. iNaturalist_teacher_104314
-->

| Model          | Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| Standard Model | 71.2             | 70.19         | 71.20           | 71.56        | Link
| Larger Model   | 72.6             | 70.93         | 72.39           | 73.11        | Link
| Teacher Model  | 72.8             | 71.02         | 72.87           | 73.27        | Link


After downloading the checkpoints, you could run evaluation by following the instructions in the test section below.