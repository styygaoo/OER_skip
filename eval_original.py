import torch
from torch.utils.data import DataLoader, Dataset
from get_vKITTI import VKITTI
from model_loader import load_model
from transforms import ToTensor
from torchvision import transforms as T
from data_processing import unpack_and_move, inverse_depth_norm, depth_norm
import time
from metrics import AverageMeter, Result


# crop = [128, 381, 45, 1196]


transformation = T.ToTensor()
trans = T.Compose([T.ToTensor()])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
maxDepth = 80
My_to_tensor = ToTensor(test=True, maxDepth=maxDepth)

# Load pre-trained model
model_original = load_model('GuideDepth', '/HOMES/yigao/KITTI_2_VKITTI/KITTI_Half_GuideDepth.pth')
model_original.to(device)
model_original.eval()
# Prepare test dataloader for TTA
# testset = VKITTI('/HOMES/yigao/KITTI/vkitti_testset_test/test', (192, 640))  #(384, 1280)
testset = VKITTI('/HOMES/yigao/KITTI/sclaing_factor_dataset/', (192, 640))  #(384, 1280)
# testset = VKITTI('/HOMES/yigao/Downloads/eval_testset/NYU_Testset', 'full')
testset_loader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)   # drop_last=True


downscale_image = T.Resize((192, 640))  # To Model resolution
########################################################################


average_meter = AverageMeter()
for i, data in enumerate(testset_loader):
    t0 = time.time()
    images, gts = data

    for b in range(images.shape[0]):
        # print(b)
        packed_data = {'image': images[b], 'depth': gts[b]}
        data = My_to_tensor(packed_data)
        image, gt = unpack_and_move(data)
        # image, gt = data['image'], data['depth']
        image = image.unsqueeze(0)
        gt = gt.unsqueeze(0)

        # image = downscale_image(image)
        # gt = downscale_image(gt)
        # print(image)
        # print(gt)
        if b >= 1:
            batched_images = torch.cat((batched_images, batched_image))
            batched_gts = torch.cat((batched_gts, batched_gt))
        else:
            batched_images = image
            batched_gts = gt
        batched_image = image
        batched_gt = gt

    batched_image.detach().cpu()
    batched_gt.detach().cpu()
    image = image.detach().cpu()
    gt = gt.detach().cpu()

    image_flip = torch.flip(batched_images, [3])
    gt_flip = torch.flip(batched_gts, [3])

    print(batched_images.shape)
    # batched_images = downscale_image(batched_images)
    # image_flip = downscale_image(image_flip)

    data_time = time.time() - t0
    t0 = time.time()
    inv_prediction, _ = model_original(batched_images)
    predictions = inverse_depth_norm(inv_prediction)

    # inv_prediction_flip, _ = model_original(image_flip)
    # prediction_flip = inverse_depth_norm(inv_prediction_flip)
    gpu_time = time.time() - t0

    upscale_depth = T.transforms.Resize(batched_gts.shape[-2:])  # To GT res

    predictions = upscale_depth(predictions)
    # prediction_flip = upscale_depth(prediction_flip)

    gt_height, gt_width = batched_gts.shape[-2:]

    # crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,
    #                       0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
    #
    # batched_gts = batched_gts[:, :, crop[0]:crop[1], crop[2]:crop[3]]
    # gt_flip = gt_flip[:, :, crop[0]:crop[1], crop[2]:crop[3]]
    # predictions = predictions[:, :, crop[0]:crop[1], crop[2]:crop[3]]
    # prediction_flip = prediction_flip[:, :, crop[0]:crop[1], crop[2]:crop[3]]

    result = Result()
    # print(batched_gts.shape)
    # print(predictions.shape)
    # print(predictions.shape)
    # print(batched_gts.shape)

    result.evaluate(predictions.data, batched_gts.data)
    average_meter.update(result, gpu_time, data_time, image.size(0))

    # result_flip = Result()
    # result_flip.evaluate(prediction_flip.data, gt_flip.data)
    # average_meter.update(result_flip, gpu_time, data_time, image.size(0))
# Report
avg = average_meter.average()
current_time = time.strftime('%H:%M', time.localtime())

print('\n*\n'
      'RMSE={average.rmse:.3f}\n'
      'MAE={average.mae:.3f}\n'
      'Delta1={average.delta1:.3f}\n'
      'Delta2={average.delta2:.3f}\n'
      'Delta3={average.delta3:.3f}\n'
      'REL={average.absrel:.3f}\n'
      'Lg10={average.lg10:.3f}\n'
      't_GPU={time:.3f}\n'.format(
    average=avg, time=avg.gpu_time))
