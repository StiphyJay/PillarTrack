from .set_tracker import SetTracker

class SMAT(SetTracker):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts = self.post_2d(batch_dict)
            return pred_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_track, tb_dict = self.decoder_head.get_loss()

        loss = loss_track
        return loss, tb_dict, disp_dict
