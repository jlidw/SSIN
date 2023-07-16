import argparse
import sys
import time
import datetime

sys.path.append('..')  # import the upper directory of the current file into the search path
from SSIN.utils.utils import *
from SSIN.Trainer import MaskedTrainer
import SSIN.utils.config as cfg


def get_default_args():
    """
    Build Default Arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hk")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--sub_out_dir", type=str, default="main")

    parser.add_argument('--mask_time', type=int, default=10, help="how many times a sequence is "
                                                                  "randomly masked to generate the training set")
    parser.add_argument('--masked_lm_prob', type=float, default=0.2, help="mask ratio")

    # Model params
    parser.add_argument('--model_type', type=str, default="SpaFormer")
    parser.add_argument('--return_attns', action='store_true', help="if return attention results")

    parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate")
    parser.add_argument('--n_layers', type=int, default=3, help="num of stacked Transformer encoder")
    parser.add_argument('--n_head', type=int, default=2, help="num of multi-head attentions")
    parser.add_argument('--d_k', type=int, default=16, help="dim of key matrix")
    parser.add_argument('--d_v', type=int, default=16, help="dim of value matrix")
    parser.add_argument('--d_model', type=int, default=16, help="dim of embedding")
    parser.add_argument('--d_inner', type=int, default=256, help="dim of inner layer")

    # Optimizer param
    parser.add_argument('--lr_mul', type=float, default=0.05, help="multiplication factor of learning rate")
    parser.add_argument('--n_warmup_steps', type=int, default=1200, help="Warm-up steps")

    # Trainer params
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
    parser.add_argument('--gpu_id', type=str, default="0", help='CUDA Visible Devices.')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    return parser


def get_data_path(args):
    base_dir = "./data"
    if args.dataset.lower() == "hk":
        dataset_dir = f"{base_dir}/HK_123_data"
        args.output_dir = f"./output/HK_output/{args.sub_out_dir}"
        args.data_prefix = "2008-2012"

        args.max_seq_len = 123
        args.max_pred_per_seq = 25  # 20%
        args.batch_size = 64
    elif args.dataset.lower() == "bw":
        dataset_dir = f"{base_dir}/BW_132_data"
        args.output_dir = f"./output/BW_output/{args.sub_out_dir}"
        args.data_prefix = "2012-2014"

        args.max_seq_len = 132
        args.max_pred_per_seq = 26  # 20%
        args.batch_size = 64
    elif args.dataset.lower() == "bay":
        dataset_dir = f"{base_dir}/PEMS-BAY"
        args.output_dir = f"./output/PEMS-BAY_output/{args.sub_out_dir}"
        args.data_prefix = "bay"

        args.mask_time = 1   # data are 5 mins, too many timestamps, just mask one time for each epoch.
        args.epochs = 2
        args.max_seq_len = 325
        args.max_pred_per_seq = 65  # 20%
        args.batch_size = 64
    else:
        raise NotImplementedError("Unsupported dataset!")

    pkl_dir = "pkl_data"

    args.train_data_path = f"{dataset_dir}/{pkl_dir}/train/{args.data_prefix}_data.pkl"
    args.test_data_path = f"{dataset_dir}/{pkl_dir}/test/{args.data_prefix}_data.pkl"


def main(args):
    s_time = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    init_seeds(args)

    get_data_path(args)

    cur_time = datetime.datetime.now()
    t_str = cur_time.strftime('%m%d-%H%M%S')
    output_dir = f"{args.output_dir}/D{args.d_k}_L{args.n_layers}_H{args.n_head}" \
                 f"_TrainOn{args.data_prefix}_TestOn{args.data_prefix}-{args.suffix}_{t_str}"

    args.output_dir = output_dir
    paths = Paths(args.output_dir)  # create out dri and sub-out dir

    save_args(args, cfg, args.output_dir)

    print("Creating Trainer")
    trainer = MaskedTrainer(args=args, out_path=paths)

    print("Training start...")
    training_time, test_time = trainer.train()  # perform training and testing

    run_time = round((time.time() - s_time) / 3600, 2)  # hour
    save_running_time(args.output_dir, run_time)
    save_running_time(args.output_dir, training_time, "All Training Time")
    save_running_time(args.output_dir, test_time, "All Testing Time")


if __name__ == "__main__":
    parser = get_default_args()
    args = parser.parse_args()

    # only setting d_k is OK; d_v=d_k=d_model
    args.d_v = args.d_k
    args.d_model = args.d_k
    main(args)


