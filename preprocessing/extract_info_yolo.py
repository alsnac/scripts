import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
import tqdm as tq

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-x",
                    "--xml_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str)
parser.add_argument("-b",
                    "--train_test",
                    help="Path to the folder where the input .xml files are stored.",
                    type=bool)
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as XML_DIR.",
                    type=str, default=None)
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.xml_dir

is_train = args.train_test
print(is_train)
#label_map = label_map_util.load_labelmap(args.labels_path)
#label_map_dict = label_map_util.get_label_map_dict(label_map)


def xml_to_csv(path):
    dicio = {'helicoverpa_armigera':0, 'spodoptera_frugiperda':1,  'dichelops_melacanthus':2,  'anticarsia_gemmatalis':3}
    xml_list = []
    list_of_images = []
    for xml_file in tq.tqdm(glob.glob(path + '/*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        filename = filename.replace('Dataset/Dataset-25-Nov-2021/Pragas/anticarsia_gemmatalis/','')
        filename = filename.replace('Dataset/Dataset-25-Nov-2021/Pragas/dichelops_melacanthus/','')
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        list_of_info = []
        list_of_info.append(filename)
        list_of_info.append(width)
        list_of_info.append(height)
        list_of_box = []
        string_out = ""
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            name_text = member.find('name').text
            value = (filename,
                     width,
                     height,
                     name_text.replace(' ','_'),
                     int(float(bndbox.find('xmin').text)),
                     int(float(bndbox.find('ymin').text)),
                     int(float(bndbox.find('xmax').text)),
                     int(float(bndbox.find('ymax').text)),
                     )
            string_out += str(dicio[name_text.replace(' ','_')])+' '
            pos_c_x = 0.5*(float(bndbox.find('xmin').text)+float(bndbox.find('xmax').text))/width
            pos_c_y = 0.5*(float(bndbox.find('ymin').text)+float(bndbox.find('ymax').text))/height
            qtd_w = (float(bndbox.find('xmax').text)-float(bndbox.find('xmin').text))/width
            qtd_h = (float(bndbox.find('ymax').text)-float(bndbox.find('ymin').text))/height
            string_out += "{:0.6f}".format(pos_c_x)+' '
            string_out += "{:0.6f}".format(pos_c_y)+' '
            string_out += "{:0.6f}".format(qtd_w)+' '
            string_out += "{:0.6f}".format(qtd_h)+'\n'
            box = []
            box.append(int(float(bndbox.find('xmin').text)))
            box.append(int(float(bndbox.find('ymin').text)))
            box.append(int(float(bndbox.find('xmax').text)))
            box.append(int(float(bndbox.find('ymax').text)))
            list_of_box.append(box)
        with open(filename[:-4]+'.txt', 'w') as f:
            f.write(string_out[:-1])
            f.close()
        list_of_info.append(list_of_box)
        list_of_images.append(list_of_info)
    #column_name = ['filename', 'width', 'height',
    #               'class', 'xmin', 'ymin', 'xmax', 'ymax']
    #xml_df = pd.DataFrame(xml_list, columns=column_name)
    return list_of_images


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def main(_):

    examples = xml_to_csv(args.xml_dir)
    import pickle
    open_file = open('box_list_train', "wb")
    pickle.dump(examples, open_file)
    open_file.close()
    #grouped = split(examples, 'filename')
    #for group in tq.tqdm(grouped):
    #    tf_example = create_tf_example(group, path)
    #    writer.write(tf_example.SerializeToString())
    #writer.close()
    #print('Successfully created the TFRecord file: {}'.format(args.output_path))
    #if args.csv_path is not None:
    #    examples.to_csv(args.csv_path, index=None)
    #    print('Successfully created the CSV file: {}'.format(args.csv_path))


if __name__ == '__main__':
    tf.app.run()
