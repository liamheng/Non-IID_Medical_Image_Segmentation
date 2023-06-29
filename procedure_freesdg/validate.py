"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python validate.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python validate.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python validate.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import csv
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.visualizer import save_images
from utils import html
from tqdm import tqdm

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # guide = True if opt.input_nc > 3 else False
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options

    metrics_name_list = opt.metrics.split(',')
    metrics_result_list = []

    for name in tqdm(os.listdir(os.path.join(opt.checkpoints_dir, opt.name))):
        if not name.endswith('.pth'):
            continue
        model.setup(opt, load_suffix=name[:-10], print_network=not opt.verbose)
        if opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if not opt.verbose and i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
        results = model.get_metric_results()
        temp_result_list = [name]
        for metric_name in metrics_name_list:
            temp_result_list.append(results[metric_name])
        metrics_result_list.append(temp_result_list)
        model.confusion_matrix.reset()

    sort_index = metrics_name_list.index(opt.sort_metric) + 1 if opt.sort_metric is not None else 1
    metrics_result_list = sorted(metrics_result_list, key=lambda x: -x[sort_index])
    with open(os.path.join(opt.results_dir, opt.name + '_analysis.csv'), 'w') as result_file:
        csv_writer = csv.writer(result_file)
        csv_writer.writerow(['pth_name'] + metrics_name_list)
        csv_writer.writerows(metrics_result_list)
    print('result file: {}'.format(opt.name + '_analysis.csv'))
    print('the best pth: {}'.format(metrics_result_list[0][0]))
