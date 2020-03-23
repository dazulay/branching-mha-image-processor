import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from numpy import long


class MultiTaskDataset(Dataset):
    
    def __init__(self,args,source_dir):
        self.filenames = glob.glob(os.path.join(source_dir, '*.npy'))
        self.report_label_offset=args.report_label_offset
        self.report_size=args.report_size
        self.report_task_offset=args.report_task_offset

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        flat = np.load(self.filenames[idx])
        saved_trace = flat[0:flat.size-self.report_size]
        trace_float = saved_trace.astype(np.float32)
        image=np.reshape(trace_float,(3,36,36))
        label=flat[saved_trace.size+self.report_label_offset:saved_trace.size+self.report_label_offset+1]
        label=label[0]
        task=flat[saved_trace.size+self.report_task_offset:saved_trace.size+self.report_task_offset+1]   
        task=task[0]
        task=task.astype(np.float32)
        indx=idx

        sample = {'image': image, 'label': label, 'task': task, 'indx': indx}
        return sample       