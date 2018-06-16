#coding:utf-8
"""
* @auther kmzk
* @date 2018-6-16
"""
import os, sys, time
import argparse
import csv, yaml
import util.yaml_utils as yaml_utils
import util.dataIO as IO

def make_coordinate_csv(out_dir, csv_name, input_size, interval, patch_side):
    '''
    * @param out_dir Directory to output
    * @param csv_name Csv file name
    * @param input_size Input volume size
    * @param patch_side Patch side
    input_size[0]=x, input_size[1]=y, input_size[2]=z
    interval[0]=x, interval[1]=y, interval[2]=z
    '''
    with open(os.path.join(out_dir, csv_name), "a", newline="") as f:
        writer = csv.writer(f)

        for z in range(int(patch_side[2]/2), input_size[2]-int(patch_side[2]/2), interval[2]):
            for y in range(int(patch_side[1]/2), input_size[1]-int(patch_side[1]/2), interval[1]):
                for x in range(int(patch_side[0]/2), input_size[0]-int(patch_side[0]/2), interval[0]):
                    writer.writerow([x,y,z])

def main():
    parser = argparse.ArgumentParser(description='Preprocessing for 3d-unet')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')
    parser.add_argument('--config_path', type=str, default='configs/base.yml',
                        help='path to config file')

    parser.add_argument('--training_list', default='configs/training_list.txt',
                        help='Path to training image list file')
    parser.add_argument('--validation_list', default='configs/validation_list.txt',
                        help='Path to validation image list file')

    parser.add_argument('--training_coordinate_list', type=str, default='configs/training_coordinate_list.csv')
    parser.add_argument('--validation_coordinate_list', type=str, default='configs/validation_coordinate_list.csv')
    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(os.path.dirname(__file__), args.config_path))))

    def read_data_list_txt(data_list_txt):
        path_pairs = []
        with open(data_list_txt) as paths_file:
            for line in paths_file:
                line = line.split()
                if not line : continue
                path_pairs.append(line[:])
        return path_pairs

    paths = read_data_list_txt(args.training_list)

    for i in paths:
        org = IO.read_mhd_and_raw(os.path.join(args.root, 'data', i[0]))
        make_coordinate_csv(out_dir = args.root,
                            csv_name = args.training_coordinate_list,
                            input_size = [org.shape[2], org.shape[1], org.shape[0]],
                            interval = [config.patch['interval'], config.patch['interval'],config.patch['interval']],
                            patch_side=[config.patch['patchside'],config.patch['patchside'],config.patch['patchside']])


    paths = read_data_list_txt(args.validation_list)

    for i in paths:
        org = IO.read_mhd_and_raw(os.path.join(args.root, 'data', i[0]))
        make_coordinate_csv(out_dir = args.root,
                            csv_name = args.validation_coordinate_list,
                            input_size = [org.shape[2], org.shape[1], org.shape[0]],
                            interval = [config.patch['interval'], config.patch['interval'],config.patch['interval']],
                            patch_side=[config.patch['patchside'],config.patch['patchside'],config.patch['patchside']])


if __name__ == '__main__':
    main()
