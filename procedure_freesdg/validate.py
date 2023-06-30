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
