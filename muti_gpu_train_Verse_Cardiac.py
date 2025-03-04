import os
import sys
import torch
import argparse
from datetime import datetime
import torch.multiprocessing as mp
import shutil
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from Verse_1_0_Cardiac.Verse_model import Verse
from tensorboardX import SummaryWriter
from datasets.data_mapper.Cardiac_data_utils import *
from tqdm import tqdm
from config.config import add_maskformer2_config
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from Verse_1_0_Cardiac.configs.Cardiac_config.Cardiac_config import add_Cardiac_config
from detectron2.config import CfgNode, LazyConfig
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from Verse_1_0_Cardiac.configs.Cardiac_config.Training_config import add_training_config
import torch.distributed as dist
import random
import time
from detectron2.config import get_cfg


def set_parse():
    parser = argparse.ArgumentParser()
    # %% set up parser
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--config_file", type=str, default='')
    # config
    parser.add_argument("--test_mode", default=False, type=bool)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    parser.add_argument('-num_workers', type=int, default=4)
    args = parser.parse_args()
    return args


def set_seed(seed):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for numpy
    np.random.seed(seed)
    # Set the seed for torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # For the cuDNN backend, set the following:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    """ Kill all child processes and destroy distributed group. """
    try:
        torch.distributed.destroy_process_group()
    except:
        pass
    os.system("pkill -f muti_gpu_train_Verse_Cardiac.py")  # Kill all processes related to training script
    sys.exit(0)

def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        add_training_config(cfg)
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


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    sys.stdout = open(log_file, 'a')
    sys.stderr = open(log_file, 'a')

def muti_gpu_train_epoch(cfg, model, train_dataloader, optimizer, scheduler, writer, epoch, rank, gpu, train_iter_num):
    epoch_loss = 0

    epoch_iterator = tqdm(
        train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True
    )
    # atexit.register(cleanup)
    if cfg.TRAINING.DIST:
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
            # torch.distributed.barrier()

    model.train()

    for batch in epoch_iterator:

        for batch_data in batch:
            batch_data['epoch'] = epoch

        loss_dict = model(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            print("loss_dict is a tensor!")
        else:
            losses = sum(loss_dict.values())

        if torch.isnan(losses).any():
            print("NaN detected in loss. Stopping training.")
            assert not torch.isnan(losses).any(), "NaN detected in loss. Training stopped."

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(f'[RANK {rank}: GPU {gpu}] EPOCH-{epoch} ITER-{train_iter_num} --- loss {losses.item()}')
        train_iter_num += 1
        if rank == 0:
            writer.add_scalar('train_iter/loss', losses, train_iter_num)
        epoch_loss += losses.item()

    scheduler.step()

    if cfg.TRAINING.DIST:
        epoch_loss_tensor = torch.tensor(epoch_loss).to(gpu)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss_tensor.item() / dist.get_world_size()

    epoch_loss /= len(train_dataloader) + 1e-12

    print(f'{cfg.TRAINING.MODEL_SAVE_PATH} ==> [GPU {gpu}] ',
          'epoch_loss: {}'.format(epoch_loss))
    if rank == 0:
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr(), epoch)
    return epoch_loss, train_iter_num


def muti_gpu_main_worker(gpu, ngpus_per_node, cfg, args):
    node_rank = int(cfg.TRAINING.NODE_RANK)
    rank = node_rank * ngpus_per_node + gpu
    # world_size = ngpus_per_node  # args.world_size
    world_size = ngpus_per_node * cfg.TRAINING.NUM_NODES
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=cfg.TRAINING.INIT_METHOD,
        rank=rank,
        world_size=world_size,
    )
    print('init_process_group finished')

    model = Verse(cfg).to(gpu)  # checkpoint for pretrained vit
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     cfg.MODEL.WEIGHTS, resume=False
    # )

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
        model.load_state_dict(checkpoint['model'], strict=True)
        print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cfg_solver = cfg.SOLVER
    if cfg_solver.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg_solver.BASE_LR,
            weight_decay=cfg_solver.WEIGHT_DECAY
        )

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=cfg_solver.WARMUP_EPOCH,
                                              max_epochs=75)
    num_epochs = cfg.TRAINING.NUM_EPOCH
    iter_num = 0
    train_dataloader = get_loader(cfg)

    start_epoch = 0
    if cfg.TRAINING.RESUME is not None:
        if os.path.isfile(cfg.TRAINING.RESUME):
            print("=> loading checkpoint '{}'".format(cfg.TRAINING.RESUME))
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(cfg.TRAINING.RESUME, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            scheduler.last_epoch = start_epoch
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAINING.RESUME, checkpoint['epoch']))

    if rank == 0:
        writer = SummaryWriter(log_dir='./tb_log/' + cfg.TRAINING.RUN_ID)
        print('Writing Tensorboard logs to ', './tb_log/' + cfg.TRAINING.RUN_ID)
    else:
        writer = None

    for epoch in range(start_epoch, num_epochs):
        with model.join():
            epoch_loss, iter_num = muti_gpu_train_epoch(cfg, model, train_dataloader, optimizer, scheduler, writer,
                                                        epoch, rank,
                                                        gpu, iter_num)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        # save the model checkpoint
        if is_main_host and epoch == 9:
            os.makedirs(cfg.TRAINING.MODEL_SAVE_PATH, exist_ok=True)
            shutil.copyfile(__file__, os.path.join(cfg.TRAINING.MODEL_SAVE_PATH,
                                                   cfg.TRAINING.RUN_ID + '_' + os.path.basename(__file__)))
        if is_main_host and epoch == 2:
            os.makedirs(cfg.TRAINING.MODEL_SAVE_PATH, exist_ok=True)
            cfg_save_path = os.path.join(cfg.TRAINING.MODEL_SAVE_PATH, 'config.json')
            with open(cfg_save_path, 'w') as cfg_file:
                json.dump(cfg, cfg_file, indent=4)

        if is_main_host and epoch > 9:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(cfg.TRAINING.MODEL_SAVE_PATH, 'latest_medsam_model.pth'))

        if is_main_host and epoch in [59, 64, 69, 74]:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
            }
            model_save_name = f"medsam_model_epoch_{epoch}.pth"
            torch.save(checkpoint, os.path.join(cfg.TRAINING.MODEL_SAVE_PATH, model_save_name))

        torch.distributed.barrier()



def main(cfg):
    # set seeds
    set_seed(2024)
    torch.cuda.empty_cache()
    if cfg.TRAINING.DIST:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        cfg.TRAINING.RUN_ID = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = os.path.join(cfg.TRAINING.WORK_DIR, cfg.TRAINING.RUN_ID)
        cfg.TRAINING.MODEL_SAVE_PATH = model_save_path
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9999'
        # ngpus_per_node = torch.cuda.device_count()
        ngpus_per_node = 4
        specific_gpus = [4,5,6,7]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, specific_gpus))

        cfg.TRAINING.NGPUS = ngpus_per_node
        print("Spwaning processces, ngpus_per_node={}".format(ngpus_per_node))
        print(f"=====> project save at {cfg.TRAINING.MODEL_SAVE_PATH}")
        mp.set_start_method('spawn')
        # mp.spawn(muti_gpu_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, args))
        try:
            mp.spawn(muti_gpu_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, args))
        except KeyboardInterrupt:
            print("\n[MAIN] Training interrupted. Cleaning up...")
            cleanup()
    else:
        # torch.multiprocessing.set_sharing_strategy('file_system')
        gpu = 0
        cfg.TRAINING.RUN_ID = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = os.path.join(cfg.TRAINING.WORK_DIR, cfg.TRAINING.RUN_ID)
        cfg.TRAINING.MODEL_SAVE_PATH = model_save_path
        print(f"=====> project save at {cfg.TRAINING.MODEL_SAVE_PATH}")
        os.makedirs(cfg.TRAINING.MODEL_SAVE_PATH, exist_ok=True)
        shutil.copyfile(__file__, os.path.join(cfg.TRAINING.MODEL_SAVE_PATH,
                                               cfg.TRAINING.RUN_ID + '_' + os.path.basename(__file__)))
        main_worker(gpu, cfg)


if __name__ == "__main__":
    args = set_parse()
    cfg = setup(args)
    main(cfg=cfg)
    sys.exit(0)
