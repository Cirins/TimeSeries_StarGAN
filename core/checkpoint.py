import os
import torch


class CheckpointIO(object):
    def __init__(self, base_dir, fname_template, data_parallel=False, **kwargs):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.fname_template = fname_template
        self.module_dict = kwargs
        self.data_parallel = data_parallel

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        fname = os.path.join(self.base_dir, fname)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            if self.data_parallel:
                outdict[name] = module.module.state_dict()
            else:
                outdict[name] = module.state_dict()
                        
        torch.save(outdict, fname)

    def load(self, step):
        fname = self.fname_template.format(step)
        fname = os.path.join(self.base_dir, fname)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname, weights_only=False)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'), weights_only=False)
            
        for name, module in self.module_dict.items():
            if self.data_parallel:
                module.module.load_state_dict(module_dict[name])
            else:
                module.load_state_dict(module_dict[name])
            # # print max and min of the weights of the model
            # for param_tensor in module.state_dict():
            #     weight = module.state_dict()[param_tensor]
            #     if 'weight' in param_tensor:  # Ensure we're looking at weight tensors
            #         max_val = weight.max().item()
            #         min_val = weight.min().item()
            #         print(f"{param_tensor} - Max Weight: {max_val}, Min Weight: {min_val}")
