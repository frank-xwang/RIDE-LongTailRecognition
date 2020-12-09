import os, sys, shutil
import torch

train_id = sys.argv[1]
epochs = sys.argv[2]
experiment_type = sys.argv[3]
norm_weights = sys.argv[4] # 0.3146 for iNaturalist, 0.6968 for ImageNet-LT
norm_weights = float(norm_weights)
test_gpu = sys.argv[5]
use_parallel = sys.argv[6] == "True" or sys.argv[6] == "1"
diversity_num = sys.argv[7]
diversity_num = int(diversity_num)

print("train_id:", train_id, "epochs:", epochs, "experiment_type:", experiment_type)

os.makedirs('saved/models/{}/{}normalized/'.format(experiment_type, train_id))
shutil.copyfile('saved/models/{}/{}/config.json'.format(experiment_type, train_id), 'saved/models/{}/{}normalized/config.json'.format(experiment_type,train_id))
shutil.copyfile('saved/models/{}/{}/checkpoint-epoch{}.pth'.format(experiment_type, train_id, epochs), 'saved/models/{}/{}normalized/checkpoint-epoch{}.pth'.format(experiment_type, train_id, epochs))

pth = torch.load('saved/models/{}/{}/checkpoint-epoch{}.pth'.format(experiment_type, train_id, epochs))

def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws

print(pth['state_dict'].keys())

def parallel(key):
    if use_parallel:
        return "module." + key
    else:
        return key

for ind in range(diversity_num):
    if diversity_num == 1:
        linear_name = "linear"
    else:
        linear_name = f"linears.{ind}"
    weights = pth['state_dict'][parallel(f'backbone.{linear_name}.weight')].cpu()
    bias = pth['state_dict'][parallel(f'backbone.{linear_name}.bias')].cpu()

    ws = pnorm(weights, norm_weights)
    bs = bias * 0

    pth['state_dict'][parallel(f'backbone.{linear_name}.weight')] = ws
    pth['state_dict'][parallel(f'backbone.{linear_name}.bias')] = bs

torch.save(pth, 'saved/models/{}/{}normalized/normalized.pth'.format(experiment_type, train_id))
os.system("python test.py -d {} -r saved/models/{}/{}normalized/normalized.pth".format(test_gpu, experiment_type, train_id))
