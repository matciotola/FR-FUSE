import os
import numpy as np
import argparse
from FR_FUSE import fr_fuse
from Utils.load_save_tools import open_tiff, extract_info, save_tiff


def main(args):

    bands_10_path = args.bands_10_path
    bands_20_path = args.bands_20_path

    out_root = args.out_dir

    bands_10 = open_tiff(bands_10_path)
    bands_20 = open_tiff(bands_20_path)
    filename = os.path.basename(bands_10_path)
    geo_info = extract_info(bands_10_path)

    fused = fr_fuse(bands_10, bands_20)

    save_tiff(np.squeeze(fused.numpy(), axis=0), out_root, filename, geo_info)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='FR-FUSE',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='FR-FUSE is an unsupervised deep learning-based for super-resolution '
                                                 'of 20m Sentinel-2 images through multi-resolution fusion',
                                     epilog='''\
    Reference: 
    A full-resolution training framework for Sentinel-2 image fusion
    Matteo Ciotola, Mario Ragosta, Giovanni Poggi, Giuseppe Scarpa

    Authors: 
    Image Processing Research Group of University of Naples Federico II ('GRIP-UNINA')
    University of Naples Parthenope

    For further information, please contact the first author by email: matteo.ciotola[at]unina.it '''
                                     )
    optional = parser._action_groups.pop()
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument("-b10", "--bands_10_path", type=str, required=True,
                               help='The path of the .tiff file which contains the 10m Sentinel-2 bands.'
                                    'For more details, please refer to the GitHub documentation.')

    requiredNamed.add_argument("-b20", "--bands_20_path", type=str, required=True,
                               help='The path of the .tiff file which contains the 20m Sentinel-2 bands.'
                                    'For more details, please refer to the GitHub documentation.')

    optional.add_argument("-o", "--out_dir", type=str, default='Results',
                          help='The directory in which save the outcome.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    main(arguments)
