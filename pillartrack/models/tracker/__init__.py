from .set_tracker import SetTracker
from .smat import SMAT

__all__ = {
    'SetTracker': SetTracker,
    'SMAT': SMAT,
}


def build_tracker(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
