import pathlib
import cv2
import kornia.utils
import torch.utils.data



class RewardingForTest(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, RGB_folder: pathlib.Path, REG_folder: pathlib.Path):
        super(RewardingForTest, self).__init__()
        self.rgb_list = [x for x in sorted(RGB_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.REG_list = [x for x in sorted(REG_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        rgb_path = self.rgb_list[index] # rgb
        REG_path = self.REG_list[index]

        name = rgb_path.name.split('.')[0]
        assert REG_path.name.split('.')[0] == rgb_path.name.split('.')[0], f"Mismatch ir:{REG_path.name} vi:{rgb_path.name}."

        # read image as type Tensor
        rgb = self.imread(path=rgb_path, flags=cv2.IMREAD_GRAYSCALE)
        reg = self.imread(path=REG_path, flags=cv2.IMREAD_GRAYSCALE)
        return (reg, rgb, name)

    def __len__(self):
        return len(self.rgb_list)


    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        im_cv = cv2.resize(im_cv, (80, 80))
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts