import sys
import argparse
import torch.multiprocessing as mp
import torch
from Verse_1_0_Cardiac.Verse_model import Verse
from datasets.data_mapper.ACDC_data_utils import *
from fvcore.common.config import CfgNode
from detectron2.utils.logger import setup_logger
from Verse_1_0_Cardiac.configs.Cardiac_config.Cardiac_config import add_Cardiac_config
from detectron2.config import CfgNode, LazyConfig
from Verse_1_0_Cardiac.configs.Cardiac_config.Testing_config import add_testing_config
import torch.distributed as dist
from utils.analysis import get_iou, get_dice
from utils.visualize_2d import draw_result_with_point
import pandas as pd
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from tqdm import tqdm
import atexit
from config.config import add_maskformer2_config
from detectron2.config import get_cfg
from fvcore.nn import FlopCountAnalysis


# import atexit

def set_parse():
    parser = argparse.ArgumentParser()
    # %% set up parser
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--config_file", type=str, default='')
    # config
    parser.add_argument("--test_mode", default=True, type=bool)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-draw_plot', type=bool, default=False)
    args = parser.parse_args()
    return args


def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        add_testing_config(cfg)
        add_Cardiac_config(cfg)
    else:
        cfg = LazyConfig.load(args.config_file)
    setup_logger(name="fvcore")
    setup_logger()
    return cfg


# Define a function to save cfg to a YAML file

def save_cfg_to_yaml(cfg, output_filename):
    with open(output_filename, "w") as f:
        f.write(cfg.dump())


def cleanup():
    dist.destroy_process_group()


def gather_results(results, world_size):
    all_results = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_results, results)
    # Merge all results dictionaries
    merged_results = {}
    for res in all_results:
        merged_results.update(res)
    return merged_results


def muti_gpu_iter_inference(gpu, ngpus_per_node, cfg, args):
    node_rank = int(cfg.TRAINING.NODE_RANK)
    rank = node_rank * ngpus_per_node + gpu
    world_size = ngpus_per_node * cfg.TRAINING.NUM_NODES
    print(f"[Rank {rank}]: Use GPU: {gpu} for inference")

    is_main_host = rank == 0
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=cfg.TRAINING.INIT_METHOD,
        rank=rank,
        world_size=world_size,
    )
    atexit.register(cleanup)
    print('init_process_group finished')

    model = Verse(cfg).to(gpu)
    # load param
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=cfg.TRAINING.BUCKET_CAP_MD
    )

    if os.path.isfile(args.resume):
        # Map model to be loaded to specified single GPU
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        model.load_state_dict(checkpoint['model'], strict=False)
        print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    else:
        raise FileNotFoundError(f"Checkpoint file '{args.resume}' not found.")

    model.training = False
    model.eval()
    # %% test
    test_iter_num = cfg.TEST.TEST_ITER_NUM
    test_dataloader = get_loader(cfg)

    epoch_iterator = tqdm(
        test_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True
    )

    iou_dict = {}
    results = {}  # Dictionary to store data for DataFrame
    results_dice = {}
    for i, batch_data in enumerate(epoch_iterator):
        bs = len(batch_data)
        processed_results_dict, point_dict = model(batch_data, mode="Testing")
        iter_num = len(processed_results_dict)

        for iter in range(iter_num):
            iou_dict[iter] = {'all': []}
            for bs_index in range(bs):
                file_name = batch_data[bs_index]['file_name']
                slice_index = batch_data[bs_index]['slice_index']

                file_name = os.path.splitext(os.path.basename(file_name))[0] + "_" + str(slice_index)
                patient_name = os.path.splitext(os.path.basename(file_name))[0]
                image = batch_data[bs_index]['image']
                gt = batch_data[bs_index]['sem_seg']
                result = processed_results_dict[iter][bs_index]
                q_index = batch_data[bs_index]['q_index']
                point = point_dict[iter][bs_index][0]
                gt_mask = (gt == (q_index)).long()
                single_result = result[0].squeeze(0)
                if torch.max(gt_mask) > 0:
                    # For interactive click mode, only test the case with valid ground truth mask.
                    if args.draw_plot:
                        work_dir = "final_plot"
                        dice = get_dice(gt_mask, single_result)
                        draw_result_with_point(image, single_result, gt_mask, work_dir, catalog=q_index,
                                               slice_name=file_name, point_tuple=point, iter_num=iter, dice=dice)

                    iou = get_iou(gt_mask, single_result)
                    dice = get_dice(gt_mask, single_result)
                    col_name = f"{file_name}_{q_index}"
                    if col_name not in results:
                        results[col_name] = [None] * 21  # Initialize with None for 20 iterations
                        results_dice[col_name] = [None] * 21
                    results[col_name][iter] = iou  # Set IOU for specific iteration
                    results_dice[col_name][iter] = dice

    gathered_results = gather_results(results, world_size)
    gathered_results_dice = gather_results(results_dice, world_size)

    # Create DataFrame from results dictionary
    if rank == 0:
        results_df = pd.DataFrame(gathered_results)
        results_df.index.name = 'Iteration'

        results_dice_df = pd.DataFrame(gathered_results_dice)
        results_dice_df.index.name = 'Iteration'

        print(f"Results DataFrame shape: {results_df.shape}")
        print(f"Number of columns in Results DataFrame: {len(results_df.columns)}")

        print(f"Results Dice DataFrame shape: {results_dice_df.shape}")
        print(f"Number of columns in Results Dice DataFrame: {len(results_dice_df.columns)}")

        iou_path = 'Verse_Cardiac_mode3_iou_ACDC_results.csv'
        dice_path = 'Verse_Cardiac_mode3_dice_ACDC_results.csv'
        results_df.to_csv(iou_path, index=True)
        results_dice_df.to_csv(dice_path, index=True)


def main(cfg, args):
    # set seeds
    torch.manual_seed(2024)
    torch.cuda.empty_cache()
    if cfg.TEST.DIST:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9999'
        ngpus_per_node = 4
        specific_gpus = [0, 1, 2, 3]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, specific_gpus))
        cfg.TRAINING.NGPUS = ngpus_per_node
        print("Spwaning processces, ngpus_per_node={}".format(ngpus_per_node))
        mp.set_start_method('spawn')
        mp.spawn(muti_gpu_iter_inference, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, args))


if __name__ == "__main__":
    args = set_parse()
    cfg = setup(args)
    main(cfg=cfg, args=args)
    sys.exit(0)
