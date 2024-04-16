"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import yaml
from pathlib import Path
from qcardia_data import DataModule
import wandb
from copy import deepcopy
import nibabel as nib
import numpy as np
import pandas as pd
import shutil
from qcardia_data.utils import sample_from_csv_by_group, data_to_file


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    config_path = Path("/home/bme001/20183502/code/msc-stijn/resources/example-config_original.yaml")

    # The config contains all the model hyperparameters and training settings for the
    # experiment. Additionally, it contains data preprocessing and augmentation
    # settings, paths to data and results, and wandb experiment parameters.
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    
    #opt.num_threads = 0   # test code only supports num_threads = 1
    #opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #config['dataloader']['train']['batch_size'] = 1
    #config['dataloader']['valid']['batch_size'] = 1

    run = wandb.init(
        project=config["experiment"]["project"],
        name=config["experiment"]["name"],
        config=config,
        save_code=True,
        mode="disabled",
    )
    dataset_name = "synthetic"
    dataset_path = Path(config['paths']['data']) / 'reformatted_data' / dataset_name
    csv_path = Path(config['paths']['data']) / 'reformatted_data' / f"{dataset_name}.csv"
    if dataset_path.exists():
        print(f"Dataset path {dataset_path} already exists, removing it now")
        shutil.rmtree(dataset_path)
    if csv_path.exists():
        print(f"CSV path {csv_path} already exists, removing it now")
        os.remove(csv_path)

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
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    meta_df = pd.DataFrame()
    meta_cols = ['subject_id', 'slice_nr', 'dataset']
    for i, data in enumerate(train_dataloader):
        unlabelled_data = next(unlabelled_iter, None)
        if unlabelled_data is None:
            print(f"Resetting unlabelled dataloader at iteration {i}")
            unlabelled_iter = iter(unlabelled_dataloader)
            unlabelled_data = next(unlabelled_iter)
        data['A'] = data[image_key]
        data['A_paths'] = data['meta_dict']['source']
        data['B'] = unlabelled_data[unlabelled_image_key]
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        generated = model.get_current_visuals()  # get image results
        outputs = generated['fake_B']
        subject_df = pd.DataFrame(unlabelled_data['meta_dict'], columns = meta_cols)
        subject_ids = []
        for i in range(outputs.shape[0]):
            subject_id = f"{unlabelled_data['meta_dict']['subject_id'][i]}_{unlabelled_data['meta_dict']['slice_nr'][i]}"
            subject_ids.append(subject_id)
            subject_path = dataset_path / subject_id
            os.makedirs(subject_path, exist_ok=True)
            image = nib.Nifti1Image(outputs[i,0,:,:].cpu().numpy(), np.eye(4))
            gt = nib.Nifti1Image(data[label_key][i,:,:,:].argmax(dim=0).cpu().float().numpy(), np.eye(4))
            nib.save(image, subject_path / f"{subject_id}_{image_key}.nii.gz")
            nib.save(gt, subject_path / f"{subject_id}_{label_key}.nii.gz")
        subject_df['SubjectID'] = subject_ids
        meta_df = pd.concat([meta_df, subject_df], ignore_index = True)
    meta_df.to_csv(csv_path)
    webpage.save()  # save the HTML

    # Update the test split file with the synthetic data
    split_path = Path(config['paths']['data']) / 'subject_splits' / config['dataset']['split_file']
    split = yaml.load(open(split_path), Loader=yaml.FullLoader)
    syn_test = sample_from_csv_by_group(csv_path, 1, "dataset", "SubjectID")
    split['test'][dataset_name] = syn_test
    data_to_file(split, split_path)
    print(F"CUT: finished generating {len(meta_df)} synthetic data for {dataset_name} and updated the split file {split_path} with the synthetic data")