import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import yaml
from pathlib import Path
from qcardia_data import DataModule
import wandb
from copy import deepcopy


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    config_path = Path("/home/bme001/20183502/code/msc-stijn/resources/example-config_original.yaml")

    # The config contains all the model hyperparameters and training settings for the
    # experiment. Additionally, it contains data preprocessing and augmentation
    # settings, paths to data and results, and wandb experiment parameters.
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    run = wandb.init(
            project=config["experiment"]["project"],
            name=config["experiment"]["name"],
            config=config,
            save_code=True,
            mode="online",
        )

    # Get the path to the directory where the Weights & Biases run files are stored.
    online_files_path = Path(run.dir)
    print(f"online_files_path: {online_files_path}")
    datasets = []
    #Split datasets if multiple different values for key_pairs are given
    if type(wandb.config["dataset"]['subsets']) == dict:
        unique_keys = []
        unique_datasets = []
        for key, value in wandb.config["dataset"]['subsets'].items():
            if value[0] not in unique_keys:
                unique_keys.extend(value)
                unique_datasets.append([key])
            else:
                unique_datasets[unique_keys.index(value[0])].append(key)
        data_config = deepcopy(wandb.config.as_dict())
        for i in range(len(unique_datasets)):
            if '=meta' in unique_keys[i][1]:
                unique_keys[i][1] = unique_keys[i][1].split('=')[0]
                data_config['dataset']['meta_only_labels'] = True
                print(f"Meta only labels datasets {unique_datasets[i]} with keys {unique_keys[i]}")
            elif str(unique_keys[i][1]).lower() in ['none', 'null', '']:
                data_config['dataset']['meta_only_labels'] = False
                unique_keys[i][1] = 'None'
                print(f"unlabelled datasets {unique_datasets[i]} with keys {unique_keys[i]}")
            else:
                data_config['dataset']['meta_only_labels'] = False
                print(f"Labelled datasets {unique_datasets[i]} with keys {unique_keys[i]}")
            data_config['dataset']['key_pairs'] = [unique_keys[i]]
            data_config['dataset']['subsets'] = unique_datasets[i]
            data_module = DataModule(data_config)
            data_module.unique_setup()
            data_module.setup()
            datasets.append(deepcopy(data_module))
    else:
        image_key, label_key = wandb.config["dataset"]["key_pairs"][
        0
        ]  # TODO: make this more general
        data_module = DataModule(wandb.config)
        data_module.unique_setup()
        data_module.setup()
        datasets.append(data_module)
        # Get the PyTorch DataLoader objects for the training and validation datasets
    unlabelled_dataloader = None
    for data in datasets:
        if data.config['dataset']['meta_only_labels'] or data.config['dataset']['key_pairs'][0][1] == 'None':
            unlabelled_image_key = data.config['dataset']['key_pairs'][0][0]
            unlabelled_dataloader = data.train_dataloader()
            unlabelled_iter = iter(unlabelled_dataloader)
        else:
            image_key, label_key = data.config["dataset"]["key_pairs"][
                0
            ] 
            train_dataloader = data.train_dataloader()
    dataset_size = len(train_dataloader)
    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(train_dataloader):  # inner loop within one epoch
            unlabelled_data = next(unlabelled_iter, None)
            if unlabelled_data is None:
                print(f"Resetting unlabelled dataloader at iteration {i}")
                unlabelled_iter = iter(unlabelled_dataloader)
                unlabelled_data = next(unlabelled_iter)
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["lge"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            data['A'] = data['lge']
            data['B'] = unlabelled_data['lge']
            data['A_paths'] = data['meta_dict']['source']
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
