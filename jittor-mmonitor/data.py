import os
import json
from jittor.dataset import Dataset, ImageFolder
from PIL import Image
import numpy as np
import jittor.transform as transform

import zipfile


class ImageFolder_self(Dataset):
    '''
    从目录中加载图像及其标签用于图像分类的数据集。

        数据集的目录结构应如下所示:
            * root/label1/img1.png
            * root/label1/img2.png
            * ...
            * root/label2/img1.png
            * root/label2/img2.png
            * ...

        参数: 
            - root (str): 包含图像和标签子目录的根目录的路径
            - json_path (str, optional): 包含要排除的图像路径的 JSON 文件路径
            - transform (callable, optional): 用于对样本进行转换的 optional 转换操作(例如, 数据增强)。默认值: None

        属性:
            - classes (list): 类名的列表
            - class_to_idx (dict): 从类名映射到类索引的字典
            - imgs (list): 包含(image_path, class_index)元组的列表

    '''
    def __init__(self, root_path, extract_to_path, json_path=None, transform=None):
        super().__init__()

        with zipfile.ZipFile(root_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)

        print(f"数据集已成功解压到 {extract_to_path}")

        print('root_path:', root_path)


        self.root = os.path.join(extract_to_path, 'train')
        self.transform = transform
        self.classes = sorted([d.name for d in os.scandir(self.root) if d.is_dir()])
        self.class_to_idx = {v: k for k, v in enumerate(self.classes)}
        self.imgs = []
        self.cache = {}
        image_exts = set(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))




        # # 加载要排除的图像列表
        # json_file_path = None
        # for file_name in os.listdir(json_path):
        #     if file_name.endswith('.json'):
        #         json_file_path = os.path.join(json_path, file_name)
        #         break  # 假设文件夹中只有一个 JSON 文件，我们找到第一个就停止

        # # 读取 JSON 文件的内容并转换为集合
        # print('json_file_path:', json_path, json_file_path)
        # excluded_images = set()
        # if json_file_path:
        #     with open(json_file_path, 'r') as json_file:
        #         # 读取 JSON 文件并转换为 Python 集合
        #         excluded_images = set(json.load(json_file))
        #     print(f"已读取 JSON 文件内容并转换为集合")
        # else:
        #     print("未找到 JSON 文件。")  
        excluded_images = np.load(json_path)

            
        # 构建图像路径和类索引列表
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root, class_name)
            for dname, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if os.path.splitext(fname)[-1].lower() in image_exts:
                        rel_path = os.path.join(class_name, fname)
                        if rel_path not in excluded_images:
                            # 跳过在 JSON 文件中列出的图像
                            continue
                        path = os.path.join(class_dir, fname)
                        self.imgs.append((path, i))
        
        print(f"Found {len(self.classes)} classes and {len(self.imgs)} images after filtering.")
        self.set_attrs(total_len=len(self.imgs))

    def __getitem__(self, k):
        if k not in self.cache:
            with open(self.imgs[k][0], 'rb') as f:
                img = Image.open(f).convert('RGB')
                self.cache[k] = img
                if self.transform:
                    img = self.transform(img)
                
                return img, self.imgs[k][1], k
        else:
            img = self.cache[k]
            if self.transform:
                img = self.transform(img)
            return img, self.imgs[k][1], k

    def __len__(self):
        return len(self.imgs)

def find_first_directory(path):
    try:
        # 获取指定路径下的所有条目
        entries = os.listdir(path)
        
        # 遍历所有条目，找到第一个不是隐藏目录且是目录的条目
        for entry in entries:
            if not entry.startswith('.') and os.path.isdir(os.path.join(path, entry)):
                return entry
        
        # 如果没有找到任何目录，返回 None
        return None
    except FileNotFoundError:
        print(f"The directory {path} does not exist.")
        return None
    except PermissionError:
        print(f"Permission denied to access the directory {path}.")
        return None

def build_dataset(args):
    normalize = transform.ImageNormalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std =[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transform.Compose([
        transform.ToTensor(),
        transform.Lambda(lambda x: (np.pad( x, ((0,0), (4,4), (4,4)), mode='reflect')).transpose(1,2,0)),
        transform.ToPILImage(),
        transform.RandomCrop(32),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        normalize,
    ])
    test_transform = transform.Compose([
        transform.ToTensor(),
        normalize
    ])

    train_loader = ImageFolder_self(root_path=os.path.join(args.cifar_10_python_path, 'CIFAR10-noise.zip'), extract_to_path=args.cifar_10_python_path, json_path=os.path.join(args.json_file_path, find_first_directory(args.json_file_path), 'sel_imgs.pkl'), transform=train_transform)
    train_loader1 = train_loader.set_attrs(num_workers=args.prefetch, batch_size=args.batch_size,shuffle=True)
    # if os.path.exists('args_params.npz'):
    # test_loader = jt.dataset.ImageFolder(root='./%s/train/' % (args.cifar_10_python_path), transform=test_transform).set_attrs(num_workers=args.prefetch, batch_size=args.batch_size,shuffle=False)

    return train_loader1



# 使用示例
if __name__ == "__main__":
    dataset_root = "./data/CIFAR10-noise.zip"
    json_file_path = "data_process_nj_results_0.0.1.zip"
    dataset = ImageFolder_self(root_path=dataset_root, extract_to_path='./', json_path=json_file_path, transform=None)

    print(len(dataset))

    dataset1 = ImageFolder(root='./train', transform=None)

    print(len(dataset1.imgs))