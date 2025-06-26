
from utils.logger import Logger as Log

from .basic_function import Compose

from .geo_aug import TestCrop, StereoPad, CenterCrop, DivisiblePad, RandomCrop, RandomCropv2

from .nongeo_aug import ChromaticAugmentation, RandomBrightnessContrastSymandAsym
from .chrometric_v2 import ChromaticAugmentationV2
from .self_aug import RightSelfAugmentation

from .basic_trans import TransposeImage, ToTensor, NormalizeImage

    

def build_transform(cfg, mean, std, stage=None):
    transform = []
    if stage is not None: # "train" "val" "test"
        assert isinstance(stage, str), "only str is supported"
        stage_stat = f"at {stage} stage"

    # crop or pad
    if cfg.get("testcrop", False):
        # used in FastACV
        transform.append(TestCrop(cfg["testcrop"], stage=stage))
    elif cfg.get("centercrop", False):
        transform.append(CenterCrop(cfg["centercrop"], stage=stage))
    elif cfg.get("randomcrop", False):
        transform.append(RandomCrop(cfg["randomcrop"], stage=stage))
    elif cfg.get("randomcropV2", False):
        transform.append(RandomCropv2(cfg["randomcropV2"], stage=stage))
    elif cfg.get("stereopad", False):
        # usually used in KITTI Dataset
        transform.append(StereoPad(cfg["stereopad"], stage=stage))
    elif cfg.get("divisiblepad", False):
        # usually used in Mid Dataset
        transform.append(DivisiblePad(cfg['divisiblepad'], stage=stage))

    # Color/Chromatic transformer
    if cfg.get('chromatic', False):
        transform.append(ChromaticAugmentation(cfg['chromatic']))
    elif cfg.get('chromaticv2', False):
        transform.append(ChromaticAugmentationV2(cfg['chromaticv2']))
    if cfg.get('bc_symasymmul', False):
        transform.append(RandomBrightnessContrastSymandAsym(cfg['bc_symasymmul']))

    # self-augmentation
    if cfg.get('selfaug', False):
        transform.append(RightSelfAugmentation(cfg['selfaug'], stage=stage))
    
    
    #basic transformer: TransposeImage --> ToTensor --> NormalizeImage
    transform.append(TransposeImage())
    transform.append(ToTensor())
    transform.append(NormalizeImage(mean, std))

    return Compose(transform)