from utils import (
    config_parser,
    seed_all,
    setup_network,
    get_loader,
    get_dataset_size,
    collect_stats)
import torch
from run_sim import optimize
from tensorboardX import SummaryWriter
import pickle
from filelock import FileLock
from real_world.realWorldEnv import RealWorldEnv
import ray
import os
import time

if __name__ == "__main__":
    args = config_parser().parse_args()
    ray.init()
    seed_all(args.seed)
    policy, optimizer, dataset_path = setup_network(args)
    criterion = torch.nn.functional.mse_loss
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    writer = SummaryWriter(logdir=args.log) if not args.eval else None
    if not os.path.exists(args.log + '/args.pkl'):
        pickle.dump(args, open(args.log + '/args.pkl', 'wb'))

    env = RealWorldEnv(replay_buffer_path=dataset_path, **vars(args))
    obs = env.reset()[0]
    dataset_size = get_dataset_size(dataset_path)
    i = get_dataset_size(dataset_path)
    start_time = time.time()
    start_size = get_dataset_size(dataset_path)
    while(True):
        with torch.no_grad():
            obs = env.step(policy.act([obs])[0])[0]
            if i > args.warmup:
                policy.decay_exploration()
        if optimizer is not None and dataset_size > args.warmup:
            if i % args.update_frequency == 0:
                policy.train()
                with FileLock(dataset_path + ".lock"):
                    for key, value_net in policy.value_nets.items():
                        optimize(
                            value_net_key=key,
                            value_net=value_net,
                            optimizer=optimizer,
                            loader=get_loader(
                                hdf5_path=dataset_path,
                                filter_fn=lambda group:
                                group.attrs['action_primitive'] == key,
                                **vars(args)),
                            criterion=criterion,
                            writer=writer,
                            num_updates=args.batches_per_update)
                policy.eval()
            checkpoint_paths = [f'{args.log}/latest_ckpt.pth']
            if i % args.save_ckpt == 0:
                checkpoint_paths.append(
                    f'{args.log}/ckpt_{policy.steps():06d}.pth')
            for path in checkpoint_paths:
                torch.save({'net': policy.state_dict(),
                            'optimizer': optimizer.state_dict()}, path)
        dataset_size = get_dataset_size(dataset_path)
        if i % 16 == 0 and dataset_size > 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            hours = float(time.time()-start_time)/3600
            print('Rate: {} datapoints/hour'.format(
                (dataset_size-start_size) / hours))

            stats = collect_stats(dataset_path)
            print('='*18 + f' {dataset_size} points ' + '='*18)
            for key, value in stats.items():
                if '_steps' in key:
                    continue
                elif 'distribution' in key:
                    if writer is not None:
                        writer.add_histogram(
                            key, value,
                            global_step=dataset_size)
                elif 'img' in key:
                    if writer is not None:
                        writer.add_image(
                            key, value,
                            global_step=dataset_size)
                else:
                    if writer is not None:
                        writer.add_scalar(
                            key, float(value),
                            global_step=dataset_size)
                    print(f'\t[{key:<36}]:\t{value:.04f}')
        i += 1
