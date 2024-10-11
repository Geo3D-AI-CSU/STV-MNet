#
# demo.py
#
import argparse
import os
import numpy as np
import time
from tqdm import tqdm

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from dataloaders import make_data_loader
from torchvision.utils import make_grid, save_image
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses
import seaborn as sns

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
ROOT=r'E:\tao\tao\pytorch-deeplab-xception-master'
def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, default=ROOT+r'\test_val\test\images', help='image to test')#required=True,
    # parser.add_argument('--out-path', type=str, required=True, help='mask image to save')
    parser.add_argument('--backbone', type=str, default='drn',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default=ROOT+r'\deeplearning\deeplab-drn\model_best.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='deeplearning',
                        choices=['pascal', 'coco', 'cityscapes','mycityscapes', 'deeplearning'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: False)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='focal',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    model_s_time = time.time()
    model = DeepLab(num_classes=args.num_classes,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = nn.DataParallel(model).cuda()
    model_u_time = time.time()
    model_load_time = model_u_time - model_s_time
    print("model load time is {}".format(model_load_time))

    cal_metrics(args, model)



def cal_metrics(args,model):
    train_loader, val_loader, test_loader, nclass = make_data_loader(args)
    evaluator = Evaluator(nclass)
    # Define Tensorboard Summary
    test_arr=[]
    model.eval()
    evaluator.reset()
    with open(os.path.join(ROOT+r'\ImageSets\Segmentation',  'test.txt'), "r") as f:
        lines = f.read().splitlines()

        for  line in enumerate(lines):
            test_arr.append(line[1])   
    tbar = tqdm(test_loader, desc='\r')
    test_loss = 0.0
    targets=[]
    preds=[]
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        dataset = test_loader.dataset
        # data_source_info = dataset.get_data_source_info()  # 假设数据集对象有这样的方法
        name = test_arr[i] # 获取第 i 个图像的路径
        print(name)
        # print(filename)
        image=image.unsqueeze(0)
        target=target.unsqueeze(0)
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)
        criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)

        loss = criterion(output, target)
        test_loss += loss.item()

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                               3)  # , normalize=False, range=(0, 255))
       
        
        # save_image(grid_image, r'E:\tao\tao\pytorch-deeplab-xception-master\test_val\test\output' + "/" + "{}_mask.png".format(name))
        u_time = time.time()
        s_time = time.time()

        img_time = u_time - s_time
        print("image:{} time: {} ".format(name, img_time))
        # save_image(grid_image, args.out_path)
        # print("type(grid) is: ", type(grid_image))
        # print("grid_image.shape is: ", grid_image.shape)
        # tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
        # Add batch sample into evaluator
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        preds.append(pred)
        targets.append(target)
        evaluator.add_batch(target, pred)
        
        
        
        

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    P = evaluator.Precision()
    R = evaluator.Recall()
    F1 = evaluator.F1_score()
    conf_matrix=evaluator.confusion_matrix
    print(conf_matrix)

    print('Test:')
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, P:{}, R:{}, F1: {}".format(Acc, Acc_class, mIoU, FWIoU,P,R,F1))
    print('Loss: %.3f' % test_loss)

    # # 计算 mAP50 和 mAP50-95
    # mAP50, mAP50_95 = evaluator.calculate_map50_and_map50_95(preds, targets)
    # print("mAP50:", mAP50)
    # print("mAP50-95:", mAP50_95)

    # 归一化混淆矩阵
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f', xticklabels=range(nclass), yticklabels=range(nclass))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # 绘制 P-R 曲线
    plt.figure(figsize=(8, 6))
    for cls in range(nclass):
        plt.plot(R[cls], P[cls], label='Class {}'.format(cls))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    

    

    
    


    

    

    # grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
    #                        3)  # , normalize=False, range=(0, 255))
    # save_image(grid_image, r'E:\tao\tao\pytorch-deeplab-xception-master\test_val\test\output' + "/" + "{}_mask.png".format(name[0:-4]))
    # u_time = time.time()
    # img_time = u_time - s_time
    # print("image:{} time: {} ".format(name, img_time))
    # save_image(grid_image, args.out_path)
    # print("type(grid) is: ", type(grid_image))
    # print("grid_image.shape is: ", grid_image.shape)


    print("image save in in_path.")



if __name__ == "__main__":
    main()

# python demo.py --in-path your_file --out-path your_dst_file
