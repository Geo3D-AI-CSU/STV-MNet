class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return r'E:\syc\StreetView\VOCdevkit\VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return r'E:\Suyingcai\StreetView\SBD\benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return r'E:\syc\StreetView\cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'deeplearning':
            return r'E:\Suyingcai\STV_MNet\code\STV_MNet\deeplab\deeplearning/'# foler that contains your dataset/
        elif dataset == 'mycityscapes':
            return r'E:/new/tao/pytorch-deeplab-xception-master/datasets/mycityscapes/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
