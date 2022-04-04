import argparse
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('input_pb',help='input pb to be converted')
   parser.add_argument('output_pbtxt',help='output pbtxt generated')
   args = parser.parse_args()
   with tf.Session() as sess:
     with tf.gfile.FastGFile(args.input_pb, 'rb') as f:
       graph_def = graph_pb2.GraphDef()
       graph_def.ParseFromString(f.read())
       #tf.import_graph_def(graph_def)
     tf.train.write_graph(graph_def, './', args.output_pbtxt, as_text=True)
