from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.add_argument('--sort_metric', type=str, default=None)
        parser.add_argument('--no_need_metric', action='store_true', help='do not need label, no metrics calculated')
        parser.add_argument('--ignore_od', action='store_true')
        # 修改这两个属性，测试时不再crop
        # parser.set_defaults(load_source_size=parser.get_default('crop_size'))
        # parser.set_defaults(load_target_size=parser.get_default('crop_size'))
        parser.add_argument('--save_variable_names', type=str, default='', help='save the images with this variable')
        parser.add_argument('--save_file_names', type=str, default='', help='save the images with this variable')
        parser.add_argument('--save_dir', type=str, default='', help='save the images with this variable')
        parser.add_argument('--require_std', action='store_true', help='require standard deviation in testing procedure')

        # used for comparison image selection
        parser.add_argument('--output_single_results', action='store_true', help='output the final result to compare and select the best sample')
        parser.add_argument('--output_dataset_name', type=str, default=None)
        parser.add_argument('--output_filename', type=str, default=None)
        parser.add_argument('--output_path', default='/home/lihaojin/data/result_cache')
        self.isTrain = False
        return parser
