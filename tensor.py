#!/usr/bin/python

import os
import tensorflow as tf
import pmt
import freezegraph

def save_graph(sess,output_path,checkpoint,checkpoint_state_name,input_graph_name,output_graph_name,outname):

    checkpoint_prefix = os.path.join(output_path,checkpoint)
    saver = tf.train.Saver(tf.all_variables())
    saver.save(sess, checkpoint_prefix, global_step=0,latest_filename=checkpoint_state_name)
    tf.train.write_graph(sess.graph.as_graph_def(),output_path,
                           input_graph_name)

    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(output_path, input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = checkpoint_prefix + "-0"
    output_node_names = outname
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(output_path, output_graph_name)
    clear_devices = False

    freezegraph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,clear_devices, "")
        

def load_graph(output_graph_path,inp,out,ckpt_path=""):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with tf.Session() as sess:
        
            with open(output_graph_path, "rb") as f:

                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

                n_input = sess.graph.get_tensor_by_name(inp)
                output = sess.graph.get_tensor_by_name(out)
                return (sess,n_input,output)

