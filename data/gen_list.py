import argparse
import os

parser = argparse.ArgumentParser(description='Data list generator')
parser.add_argument('mode', help='"data" or "labeledData"')
parser.add_argument('-o', '--output_file', help='Output file name', default='output.lst')
parser.add_argument('-a', '--append', help='Output will have been appended to the specified file', action='store_true')
parser.add_argument('-d', '--data_dir', help='Directory which contains data', required=True)
parser.add_argument('-l', '--labels_dir', help='Directory which contains labels', default='')


def _main(args):
    mode = args.mode
    output_file = os.path.expanduser(args.output_file)
    data_dir = os.path.expanduser(args.data_dir)
    labels_dir = os.path.expanduser(args.labels_dir)
    append_flag = args.append

    if not mode in ['data', 'labeledData']:
        print('ERROR: Mode must be either "data" or "labeledData"')
        exit()

    if not os.path.exists(data_dir):
        print('ERROR: No such file or directory: {}'.format(data_dir))
        exit()

    if mode == 'labeledData':
        if len(labels_dir) <= 0:
            print('ERROR: Following argument is required: -l/--labels_dir')
            exit()
        elif not os.path.exists(labels_dir):
            print('ERROR: No such file or directory: {}'.format(labels_dir))
            exit()

    if not '.' in output_file:
        output_file += '.lst'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if mode == 'data':
        gen_data_list(data_dir, output_file, append_flag)
    else:
        gen_data_label_list(data_dir, labels_dir, output_file, append_flag)


def gen_data_list(data_dir, output_file, append_flag):
    with open(output_file, 'a' if append_flag else 'w') as output:
        for data_path in [os.path.join(data_dir, file) for file in sorted(os.listdir(data_dir))]:
            if os.path.isfile(data_path):
                output.write(data_path + '\n')


def gen_data_label_list(data_dir, labels_dir, output_file, append_flag):
    with open(output_file, 'a' if append_flag else 'w') as output:
        data_list = [os.path.join(data_dir, file) for file in sorted(os.listdir(data_dir))]
        label_list = [os.path.join(labels_dir, file) for file in sorted(os.listdir(labels_dir))]

        for data_path, label_path in zip(data_list, label_list):
            if os.path.isfile(data_path) and os.path.isfile(label_path):
                output.write(data_path + ' ' + label_path + '\n')


if __name__ == '__main__':
    _main(parser.parse_args())
