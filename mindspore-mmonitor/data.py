import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore import dtype as mstype
image_size = 32  
workers = 4  
batch_size = 256  
def create_dataset_cifar10(dataset_dir, usage, resize, batch_size, workers):

    data_set = ds.Cifar10Dataset(dataset_dir=dataset_dir,
                                 usage=usage,
                                 num_parallel_workers=workers,
                                 shuffle=True)

    trans = []
    if usage == "train":
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        vision.Resize(resize),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]

    target_trans = transforms.TypeCast(mstype.int32)

    # 数据映射操作
    data_set = data_set.map(operations=trans,
                            input_columns='image',
                            num_parallel_workers=workers)

    data_set = data_set.map(operations=target_trans,
                            input_columns='label',
                            num_parallel_workers=workers)

    # 批量操作
    data_set = data_set.batch(batch_size)

    return data_set

def build_dataset(data_dir):
    dataset_train = create_dataset_cifar10(dataset_dir=data_dir,
                                        usage="train",
                                        resize=image_size,
                                        batch_size=batch_size,
                                        workers=workers)
    step_size_train = dataset_train.get_dataset_size()

    dataset_val = create_dataset_cifar10(dataset_dir=data_dir,
                                        usage="test",
                                        resize=image_size,
                                        batch_size=batch_size,
                                        workers=workers)
    step_size_val = dataset_val.get_dataset_size()
    return dataset_train,dataset_val,step_size_train,step_size_val