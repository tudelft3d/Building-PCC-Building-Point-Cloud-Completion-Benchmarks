##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
import argparse
import os
import numpy as np
import cv2
import sys
import shutil
import open3d as o3d
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose
from tqdm import tqdm
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    assert m != 0
    return pc, centroid, m

def pc_norm_with_centroid_and_scale(pc, centroid, m):
    """ pc: NxC, return NxC """
    pc = pc - centroid
    pc = pc / m
    return pc

def pc_denorm(pc, centroid, m):
    pc = pc * m
    pc = pc + centroid
    return pc

def chamfer_distance(points1, points2):
    # Calculate the pairwise Euclidean distances between points in the two sets
    distances1 = np.sqrt(np.sum((points1[:, np.newaxis] - points2) ** 2, axis=-1))
    distances2 = np.sqrt(np.sum((points2[:, np.newaxis] - points1) ** 2, axis=-1))

    # Compute the minimum distance from each point in set 1 to set 2
    min_distances1 = np.min(distances1, axis=1)

    # Compute the minimum distance from each point in set 2 to set 1
    min_distances2 = np.min(distances2, axis=1)

    # Chamfer distance is the sum of the average minimum distances in both directions
    chamfer_dist_L1 = (np.mean(np.sqrt(min_distances1)) + np.mean(np.sqrt(min_distances2))) / 2
    chamfer_dist_L2 = np.mean(min_distances1) + np.mean(min_distances2)

    return chamfer_dist_L1, chamfer_dist_L2

def chamfer_distance_cuda(points1, points2):
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    points1_tensor = torch.tensor(points1, dtype=torch.float32).cuda()
    points2_tensor = torch.tensor(points2, dtype=torch.float32).cuda()
    chamfer_dist_L1 = ChamferDisL1(points1_tensor, points2_tensor)
    chamfer_dist_L2 = ChamferDisL2(points1_tensor, points2_tensor)

    return chamfer_dist_L1.item(), chamfer_dist_L2.item()

def get_f_score(pred_pc, gt_pc, th=0.01):
    pred = o3d.geometry.PointCloud()
    pred.points = o3d.utility.Vector3dVector(pred_pc)
    gt = o3d.geometry.PointCloud()
    gt.points = o3d.utility.Vector3dVector(gt_pc)

    dist1 = pred.compute_point_cloud_distance(gt)
    dist2 = gt.compute_point_cloud_distance(pred)

    recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    result = 2 * recall * precision / (recall + precision) if recall + precision else 0.
    return result

def inference_kernel(model, pc_ndarray, args, pc_path):
    dense_points = []
    if pc_ndarray.shape[0] <= 2048:
        transform = Compose([{
            'callback': 'UpSamplePoints',
            'parameters': {
                'n_points': 2048
            },
            'objects': ['input']
        }, {
            'callback': 'ToTensor',
            'objects': ['input']
        }])

        pc_ndarray_normalized = transform({'input': pc_ndarray})
        ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
        dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    else:
        transform = Compose([{
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': 2048
            },
            'objects': ['input']
        }, {
            'callback': 'ToTensor',
            'objects': ['input']
        }])

        pc_ndarray_normalized = transform({'input': pc_ndarray})
        ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
        dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    # else:
    #     pc_ndarray_normalized = torch.tensor(pc_ndarray, dtype=torch.float32)
    #     ret = model(pc_ndarray_normalized.unsqueeze(0).to(args.device.lower()))
    #     dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    return dense_points

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='', help = 'yaml config file')
    parser.add_argument('--model_checkpoint', type=str, default='', help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--truth_root', type=str, default='', help='truth root')
    parser.add_argument('--model_3d_root', type=str, default='', help='3d model root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')
    parser.add_argument('--out_pc_root', type=str, default='', help='root of the output pc file. ')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--save_predict',type=bool, default=True, help='whether to save ply of predict point cloud')
    parser.add_argument('--save_merged', type=bool,  default=True, help='whether to save merged point cloud (merge predict and input)')
    parser.add_argument('--save_vis_img', type=bool, default=True, help='Default not saving the visualization images.')
    parser.add_argument('--copy_original_files', type=bool, default=True, help='whether to copy original point cloud, ground truth and 3D model to the inference folder')
    parser.add_argument('--evaluation_only', type=bool, default=False, help='Perform evaluation on predicted data.')
    parser.add_argument('--with_recompute', type=bool, default=False, help='Perform evaluation and recompute chamfer distance, only valid when evaluation_only is true.')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config != ''
    assert args.model_checkpoint != ''
    assert (args.pc != '') or (args.pc_root != '')

    return args

def inference_single(model, pc_path, args, model_config, avg_f_score, avg_chamfer_L1, avg_chamfer_L2, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path

    # read single point cloud
    pc_ndarray = IO.get(pc_file).astype(np.float64)

    if args.truth_root != '':
        truth_file = os.path.join(args.truth_root, pc_path.split('.')[0] + ".ply")
        # read truth point cloud for normalization
        gt_ndarray = IO.get(truth_file).astype(np.float64)

    # transform it according to the model
    centroid = 0
    m = 0
    if model_config.dataset.train._base_['NAME'] == 'ShapeNet':
        # normalize it to fit the model on ShapeNet-55/34
        centroid = np.mean(pc_ndarray, axis=0)
        pc_ndarray = pc_ndarray - centroid
        m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
        pc_ndarray = pc_ndarray / m
    elif model_config.dataset.train._base_['NAME'] == 'BuildingNL':
        if args.truth_root != '':
            _, centroid, m = pc_norm(gt_ndarray)
            pc_ndarray_trans = pc_norm_with_centroid_and_scale(pc_ndarray, centroid, m)
        else:
            pc_ndarray_trans, centroid, m = pc_norm(pc_ndarray)

    dense_points = inference_kernel(model, pc_ndarray_trans, args, pc_path)

    if args.out_pc_root != '':
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        os.makedirs(target_path, exist_ok=True)

    if model_config.dataset.train._base_['NAME'] == 'ShapeNet':
        # denormalize it to adapt for the original input
        dense_points = dense_points * m
        dense_points = dense_points + centroid
    elif model_config.dataset.train._base_['NAME'] == 'BuildingNL':
        dense_points = pc_denorm(dense_points, centroid, m)

    if args.truth_root != '':
        #chamfer_L1, chamfer_L2 = chamfer_distance(dense_points, truth_ndarray_trans)
        chamfer_L1, chamfer_L2 = chamfer_distance_cuda(dense_points, gt_ndarray)
        f_score = get_f_score(dense_points, gt_ndarray, 0.1)
        eval_str = "L1: " + str(chamfer_L1) + ", L2: " + str(chamfer_L2) + ", F_score: " + str(f_score)
        #print("L1: " + str(chamfer_L1) + ", L2: " + str(chamfer_L2) + ", F_score: " + str(f_score))
        eval_path = os.path.join(target_path,  pc_path.split('.')[0] + "_eval.txt")
        with open(eval_path, 'w') as file:
            file.write(eval_str)
        file.close()

        avg_chamfer_L1 = avg_chamfer_L1 + chamfer_L1
        avg_chamfer_L2 = avg_chamfer_L2 + chamfer_L2
        avg_f_score = avg_f_score + f_score

    np.save(os.path.join(target_path, pc_path.split('.')[0] +'_predict.npy'), dense_points)
    if args.save_vis_img:
        input_img = misc.get_ptcloud_img(pc_ndarray)
        dense_img = misc.get_ptcloud_img(dense_points)
        cv2.imwrite(os.path.join(target_path, pc_path.split('.')[0] + '_input.jpg'), input_img)
        cv2.imwrite(os.path.join(target_path, pc_path.split('.')[0] + '_predict.jpg'), dense_img)

    if args.save_predict:
        IO._write_ply(os.path.join(target_path, pc_path.split('.')[0] + '_predict.ply'),dense_points)

    if args.save_merged:
        merged_points = np.vstack((pc_ndarray, dense_points))
        IO._write_ply(os.path.join(target_path, pc_path.split('.')[0] + '_merged.ply'), merged_points)

    if args.copy_original_files:
        pc_orig_target_path = os.path.join(target_path, pc_path.split('.')[0] + "_partial.las")
        shutil.copy(pc_file, pc_orig_target_path)

        if args.truth_root != '':
            truth_target_path = os.path.join(target_path, pc_path.split('.')[0] + "_complete.ply")
            shutil.copy(truth_file, truth_target_path)

        if args.model_3d_root != '':
            model_3d = os.path.join(args.model_3d_root, pc_path.split('.')[0] + ".ply")
            model_3d_target_path = os.path.join(target_path, pc_path.split('.')[0] + "_model.ply")
            shutil.copy(model_3d, model_3d_target_path)

    return avg_f_score, avg_chamfer_L1, avg_chamfer_L2

def main():
    #torch.backends.cudnn.enabled = False
    # # torch.backends.cudnn.benchmark = True
    # # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # # torch.cuda.set_device(0)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    args = get_args()
    # init config
    model_config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(model_config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    #base_model = base_model.cuda()

    avg_f_score = 0
    avg_chamfer_L1 = 0
    avg_chamfer_L2 = 0

    if args.evaluation_only:
        if args.pc_root != '':
            pc_file_list = os.listdir(args.pc_root)
            file_i = 0
            valid_file_i = 0
            non_valid_files = []
            for pc_file in pc_file_list:
                print("Process " + str(file_i) + " / " + str(len(pc_file_list)) + " file: "+ str(os.path.splitext(pc_file)[0]))
                chamfer_L1 = 0
                chamfer_L2 = 0
                f_score = 0
                target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_file)[0])
                if args.with_recompute:
                    pred_path = os.path.join(target_path, pc_file.split('.')[0] + '_predict.ply')
                    dense_points = IO.get(pred_path).astype(np.float64)
                    gt_path = os.path.join(target_path, pc_file.split('.')[0] + "_complete.ply")
                    complete_points = IO.get(gt_path).astype(np.float64)
                    #chamfer_L1, chamfer_L2 = chamfer_distance(dense_points, complete_points)
                    chamfer_L1, chamfer_L2 = chamfer_distance_cuda(dense_points, complete_points)
                    f_score = get_f_score(dense_points, complete_points, 0.1)

                    eval_str = "L1: " + str(chamfer_L1) + ", L2: " + str(chamfer_L2) + ", F_score: " + str(f_score)
                    eval_path = os.path.join(target_path, pc_file.split('.')[0] + "_eval.txt")
                    with open(eval_path, 'w') as file:
                        file.write(eval_str)
                    file.close()

                else:
                    eval_path = os.path.join(target_path, pc_file.split('.')[0] + "_eval.txt")
                    if os.path.exists(eval_path):
                        with open(eval_path, 'r') as file:
                            content = file.read()
                        try:
                            chamfer_L1 = float(content.split('L1: ')[1].split(',')[0])
                            chamfer_L2 = float(content.split('L2: ')[1].split(',')[0])
                            f_score = float(content.split('F_score: ')[1])
                        except (ValueError, IndexError) as e:
                            print('Error parsing values from the file:', e)

                print(" - L1: " + str(chamfer_L1) + ", L2: " + str(chamfer_L2) + ", F_score: " + str(f_score))
                if chamfer_L1 != np.nan and chamfer_L2 != np.nan and f_score != np.nan and \
                        chamfer_L1 < 100000 and chamfer_L2 < 10000000 and f_score <= 1.0:
                    avg_chamfer_L1 = avg_chamfer_L1 + chamfer_L1
                    avg_chamfer_L2 = avg_chamfer_L2 + chamfer_L2
                    avg_f_score = avg_f_score + f_score
                    valid_file_i = valid_file_i + 1
                else:
                    non_valid_files.append(os.path.splitext(pc_file)[0])

                file_i = file_i + 1

            avg_f_score /= valid_file_i
            avg_chamfer_L1 /= valid_file_i
            avg_chamfer_L2 /= valid_file_i

            print("L1: " + str(avg_chamfer_L1) + ", L2: " + str(avg_chamfer_L2) + ", F_score: " + str(avg_f_score) + ", over " + str(valid_file_i) + " valid files")
            eval_str = "L1: " + str(avg_chamfer_L1) + ", L2: " + str(avg_chamfer_L2) + ", F_score: " + str(avg_f_score)
            eval_path = os.path.join(args.out_pc_root,  "final_evaluation.txt")
            with open(eval_path, 'w') as file:
                file.write(eval_str)
            file.close()

            non_valid_path = os.path.join(args.out_pc_root,  "non_valid_files.txt")
            with open(non_valid_path, 'w') as file:
                for item in non_valid_files:
                    file.write("%s\n" % item)
            file.close()
    else:
        if args.pc_root != '':
            pc_file_list = os.listdir(args.pc_root)
            for pc_file in tqdm(pc_file_list, desc="Processing Files"):
                target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_file)[0])
                eval_path = os.path.join(target_path, pc_file.split('.')[0] + "_eval.txt")
                pred_path = os.path.join(target_path, pc_file.split('.')[0] + '_predict.ply')
                comple_path = os.path.join(target_path, pc_file.split('.')[0] + "_complete.ply")
                if os.path.exists(eval_path) and os.path.exists(pred_path) and os.path.exists(comple_path):
                    with open(eval_path, 'r') as file:
                        content = file.read()
                    try:
                        chamfer_L1 = float(content.split('L1: ')[1].split(',')[0])
                        chamfer_L2 = float(content.split('L2: ')[1].split(',')[0])
                        f_score = float(content.split('F_score: ')[1])

                        avg_chamfer_L1 = avg_chamfer_L1 + chamfer_L1
                        avg_chamfer_L2 = avg_chamfer_L2 + chamfer_L2
                        avg_f_score = avg_f_score + f_score
                    except (ValueError, IndexError) as e:
                        print('Error parsing values from the file:', e)
                else:
                    avg_f_score, avg_chamfer_L1, avg_chamfer_L2 = inference_single(base_model, pc_file, args, model_config, avg_f_score, avg_chamfer_L1, avg_chamfer_L2, root=args.pc_root)

            avg_f_score /= len(pc_file_list)
            avg_chamfer_L1 /= len(pc_file_list)
            avg_chamfer_L2 /= len(pc_file_list)

            print("L1: " + str(avg_chamfer_L1) + ", L2: " + str(avg_chamfer_L2) + ", F_score: " + str(avg_f_score))
            eval_str = "L1: " + str(avg_chamfer_L1) + ", L2: " + str(avg_chamfer_L2) + ", F_score: " + str(avg_f_score)
            eval_path = os.path.join(args.out_pc_root,  "final_evaluation.txt")
            with open(eval_path, 'w') as file:
                file.write(eval_str)
            file.close()

        else:
            inference_single(base_model, args.pc, args, model_config, avg_f_score, avg_chamfer_L1, avg_chamfer_L2)

if __name__ == '__main__':
    main()