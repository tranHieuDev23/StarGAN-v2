import os
import torch


class CheckpointHandler:
    def __init__(self, filename_template: str, kwargs: dict):
        os.makedirs(os.path.dirname(filename_template), exist_ok=True)
        self.filename_template = filename_template
        self.module_dict = kwargs

    def register(self, kwargs: dict):
        self.module_dict.update(kwargs)

    def save(self, step: int):
        filename = self.filename_template.format(step)
        print("Saving checkpoint to {}...".format(filename))
        output = {}
        for name, module in self.module_dict.items():
            output[name] = module.state_dict()
        torch.save(output, filename)

    def load(self, step: int):
        filename = self.filename_template.format(step)
        if (not os.path.exists(filename)):
            raise RuntimeError("File {} doesn't exist!".format(filename))
        if (torch.cuda.is_available()):
            module_dict = torch.load(filename)
        else:
            module_dict = torch.load(
                filename, map_location=torch.device("cpu"))
        for name, module in self.module_dict.items():
            module.load_state_dict(module_dict[name])
