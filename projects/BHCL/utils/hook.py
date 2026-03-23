
import torch.distributed as dist

from mmengine.runner.runner import Runner, HOOKS, Hook


@HOOKS.register_module()
class SetEpochHook(Hook):

    def before_train_epoch(self, runner: Runner):
        if dist.is_initialized():
            model = runner.model.module
        else:
            model = runner.model
        if model.roi_head.hcl_loss is not None:
            for i in range(len(model.roi_head.hcl_loss)):
                model.roi_head.hcl_loss[i].set_epoch(runner.epoch + 1)
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f'========== Current Epoch: {model.roi_head.hcl_loss[-1].current_epoch} ==========')
