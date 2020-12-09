## FAQ
### How to select device?
Use `-d device` argument.

### iNaturalist seems to require me to pay to AWS?
It seems so, but it is out of the control of us.

### I get lower/higher accuracy than your reported one.
Since this is an re-implementation upon a pytorch template for clearity (the original codebase has too many unrelated functionalities and has some "hacks" in coding), although we tried our best in reimplementing what is described, we still could not guarantee 100% that the implementation is the same. However, **we ran experiments on all three datasets with all common settings and found the final output to be at least as high as reported most of the time**. If you get higher accuracy, then you are good since we also observe this sometimes and the reported ones are average ones, not the best ones. If you get lower accuracy and you think it's the issue with implementation, please contact us (see README and the contact method).

### It says `config.json` is not found when I load the checkpoint.
The checkpoint needs to be placed with config.json.

### It does not use the device I specified.
Make sure that `n_gpus` equals to number of your devices you specified.

### How to resume training?
```
python train.py -r path_to_checkpoint
```

The config will be automatically saved with checkpoint and loaded. If you encountered problems with resuming, please contact the code author.

### Does it support FP16 training?
If you set `fp16` in utils, it will enable fp16 training. However, this is susceptible to change (and may not work on all settings or models) and please double check if you are using it since we don't plan to focus on this part if you request help. Only some models work (see `autograd` in the code). We do not plan to provide support on this because it is not within our focus (just for faster training and less memory requirement).

### How to tune hyper-parameters in this project
See `train.py` for available options to tune. Set `collaborative_loss` is 1 to enable and 0 to disable.