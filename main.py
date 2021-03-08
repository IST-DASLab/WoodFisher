"""
File for main compression experiment
Main usage: invoke from bash script in run/ folder
accompanied by the corresponding config from config/ folder

"""
import logging
import time
import os
from options import get_parser
from policies import Manager


def setup_logging(args):

    from importlib import reload
    reload(logging)

    # attrs independent of checkpoint restore
    args.logging_level = getattr(logging, args.logging_level.upper())

    import datetime
    ts = time.time()
    run_id = str(datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H-%M-%S_%f'))

    run_id += "_{}".format(args.sweep_id) # to easily distinguish the experiment in sweep
    args.exp_dir = os.path.join(args.experiment_root_path, args.exp_name)
    args.run_dir = os.path.join(args.exp_dir, run_id)
    args.run_id = run_id
    # Make directories
    os.makedirs(args.run_dir, exist_ok=True)

    log_file_path = os.path.join(args.run_dir, 'log')
    # in append mode, because we may want to restore checkpoints
    logging.basicConfig(filename=log_file_path, filemode='a',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=args.logging_level)

    console = logging.StreamHandler()
    console.setLevel(args.logging_level)
    logging.getLogger('').addHandler(console)

    logging.info(f'Started logging run {run_id} of experiment {args.exp_name}, '+\
        f'saving checkpoints every {args.checkpoint_freq} epoch')

    return args


if __name__ == "__main__":
    args = get_parser()
    args = setup_logging(args)
    manager = Manager(args)
    manager.run()


