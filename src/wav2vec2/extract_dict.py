import torch
import sys

model = torch.load(sys.argv[1])
final_path = sys.argv[2]
model['task_state']['target_dictionary'].save(final_path + '/' +  'dict.ltr.txt')