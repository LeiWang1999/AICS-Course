import argparse
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('input_pbtxt',help='input pbtxt to be converted')
   parser.add_argument('output_pb',help='output pb generated')
   args = parser.parse_args()
   with tf.Session() as sess:
     with tf.gfile.FastGFile(args.input_pbtxt, 'rb') as f:
       graph_def = graph_pb2.GraphDef()
       new_graph_def=text_format.Merge(f.read(), graph_def)
     tf.train.write_graph(new_graph_def, './', args.output_pb, as_text=False)
