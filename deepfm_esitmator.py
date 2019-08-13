# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import random as rn
import tensorflow as tf

from tensorflow.python.keras.layers import Input, Embedding, Dense
from tensorflow.python.keras.layers import Concatenate, Reshape, Dot
from tensorflow.python.keras.layers import BatchNormalization, Dropout
from tensorflow.python.keras.layers import Lambda, Multiply
from tensorflow.python.keras.regularizers import l2 as l2_reg
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.contrib.distribute.python.mirrored_strategy import MirroredStrategy

from tensorflow.python.lib.io import file_io

from tensorflow.contrib.data.python.ops.interleave_ops import parallel_interleave

import os
import time
import sys
import json

import threading
import multiprocessing

try:
    import queue
except ImportError:
    import Queue as queue

np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)

FLAGS = None


class MySumLayer(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MySumLayer, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            if K.ndim(x) != K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0, 2, 1])
            x = x * mask
            if K.ndim(x) == 2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)
        else:
            if K.ndim(x) == 2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        if len(output_shape) == 1:
            output_shape.append(1)
        return tuple(output_shape)

    def get_config(self):
        base_config = super(MySumLayer, self).get_config()
        base_config['axis'] = self.axis
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DeepFMEstimator:
    def __init__(self, feature_size, field_size, k=8,
                 dropout_keep_fm=[1.0, 1.0],
                 deep_layers=[32, 32, 1],
                 dropout_keep_deep=[0.5, 0.5, 0.5],
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type='adam',
                 verbose=1, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type='logloss', eval_metric='auc',
                 l2=0.0, l2_fm=0.0,
                 log_dir='./output', bestModelPath='./output/keras.model',
                 greater_is_better=True,
                 num_workers=2
                 ):
        assert (use_fm or use_deep)
        assert loss_type in ['logloss', 'mse', 'ranking_logloss'], \
            'loss_type can be either "logloss" for classification task or "mse" for regression task or "ranking_logloss" for ranking task'

        self.feature_size = feature_size  # denote as M, size of the feature dictionary
        self.field_size = field_size  # denote as F, size of the feature fields
        self.k = k  # denote as K, size of the feature embedding

        self.dropout_keep_fm = dropout_keep_fm
        self.deep_layers = deep_layers
        self.dropout_keep_deep = dropout_keep_deep

        self.use_fm = use_fm
        self.use_deep = use_deep

        self.verbose = verbose
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric

        self.l2 = l2
        self.l2_fm = l2_fm

        self.bestModelPath = bestModelPath
        self.log_dir = log_dir

        self.greater_is_better = greater_is_better
        self.num_workers = num_workers

        # self.deepfm_model_fn()

    def deepfm_model_fn(self, features, labels, mode):

        feat_index = features['feat_index']
        feat_value = features['feat_value']

        feat_index = tf.reshape(feat_index, [-1, self.field_size])
        feat_value = tf.reshape(feat_value, [-1, self.field_size])

        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # self.feat_index = Input(shape=(self.field_size,), name='feat_index', dtype=tf.int64)  # None*F
        # self.feat_value = Input(shape=(self.field_size,), name='feat_value', dtype=tf.float32)  # None*F

        embeddings = Embedding(self.feature_size, self.k, name='feature_embeddings',
                               embeddings_initializer='truncated_normal',
                               embeddings_regularizer=l2_reg(self.l2_fm))(feat_index)  # None*F*k
        # embeddings = tf.keras.layers.BatchNormalization()(embeddings)
        tmp_feat_value = Reshape((self.field_size, 1))(feat_value)
        embeddings = Multiply()([embeddings, tmp_feat_value])  # None*F*8

        ###----first order------######
        y_first_order = Embedding(self.feature_size, 1, name='feature_bias',
                                  embeddings_initializer='truncated_normal',
                                  embeddings_regularizer=l2_reg(self.l2))(feat_index)  # None*F*1
        y_first_order = Multiply()([y_first_order, tmp_feat_value])  # None*F*1
        y_first_order = MySumLayer(axis=1)(y_first_order)  # None*1
        # self.y_first_order = Dropout(self.dropout_keep_fm[0], seed=self.seed)(self.y_first_order)  # None*1

        ###------second order term-------###
        # sum_square part
        summed_feature_emb = MySumLayer(axis=1)(embeddings)  # None*k
        summed_feature_emb_squred = Multiply()([summed_feature_emb, summed_feature_emb])  # None*k
        # square_sum part
        squared_feature_emb = Multiply()([embeddings, embeddings])
        squared_sum_feature_emb = MySumLayer(axis=1)(squared_feature_emb)  # None*k

        # second order
        y_second_order = Lambda(lambda x: x[0] - x[1])(
            [summed_feature_emb_squred, squared_sum_feature_emb])  # None*k
        y_second_order = MySumLayer(axis=1)(y_second_order)  # None*1
        y_second_order = Lambda(lambda x: x * 0.5)(y_second_order)  # None*k

        ##deep
        y_deep = Reshape((self.field_size * self.k,))(embeddings)

        for i in range(0, len(self.deep_layers)):
            y_deep = Dense(self.deep_layers[i], activation='relu')(y_deep)
            y_deep = Dropout(self.dropout_keep_deep[i], seed=self.seed)(y_deep)  # None*32

        # deepFM
        if self.use_fm and self.use_deep:
            concat_y = Concatenate()([y_first_order, y_second_order, y_deep])
        elif self.use_fm:
            concat_y = Concatenate()([y_first_order, y_second_order])
        elif self.use_deep:
            concat_y = y_deep

        logits = Dense(1, name='output')(concat_y)  # None*1
        y = tf.sigmoid(logits)

        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                'predict': tf.estimator.export.PredictOutput(y)
            }
            spec = tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=y,
                                              export_outputs=export_outputs)
        else:

            cross_entropy = tf.losses.sigmoid_cross_entropy(labels, logits=logits)

            # Reduce the cross-entropy batch-tensor to a single number
            # which can be used in optimization of the neural network.
            loss = tf.reduce_mean(cross_entropy)

            # Define the optimizer for improving the neural network.
            if self.optimizer_type == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer_type == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)

            # Get the TensorFlow op for doing a single optimization step.
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            # Define the evaluation metrics,
            # in this case the classification accuracy.
            if self.eval_metric == 'auc':
                metrics = {
                    'auc': tf.metrics.auc(labels, y)
                }
            else:
                metrics = {
                    "accuracy": tf.metrics.accuracy(labels, y)
                }

            # Wrap all of this in an EstimatorSpec.
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics)

        return spec


def oss_odps_table_generator(table_name):
    all_files = list_odps_table_oss_files(table_name)
    for fpath in all_files:
        f = file_io.FileIO(fpath, mode="r")
        line = f.readline()
        while line:
            yield line.strip()
            line = f.readline()
        f.close()


def list_odps_table_oss_files(table_name):
    table_data_meta = os.path.join(FLAGS.buckets, table_name, '.odps/.meta')
    print(table_data_meta)

    meta_str = file_io.read_file_to_string(table_data_meta)
    print('meta_str:', meta_str)

    meta_dict = json.loads(meta_str)
    table_data_dirs = meta_dict['dirs']
    table_data_dirs = [os.path.join(FLAGS.buckets, table_name, '.odps', d) for d in table_data_dirs]

    all_files = []
    for data_dir in table_data_dirs:
        print("data_dir:", data_dir)
        tmp_files = file_io.list_directory(data_dir)
        for fname in tmp_files:
            all_files.append(os.path.join(data_dir, fname))
    return all_files


def get_oss_table_line_count(table_name):
    all_files = list_odps_table_oss_files(table_name)
    line_count = 0
    for fpath in all_files:
        f = file_io.FileIO(fpath, mode="r")
        lines = f.readlines()
        line_count += len(lines)
    return line_count


class OssDataFeeder:

    def __init__(self, table_name):
        filenames = list_odps_table_oss_files(table_name)
        filenames.sort()
        self.filenames = []
        for i in range(len(filenames)):
            if FLAGS.task_index == i % FLAGS.worker_num:
                self.filenames.append(filenames[i])

    def line2xy(self, line):
        line = line.split(' ')
        y = int(line[0])
        Xi = [int(x.split(':')[0]) for x in line[1:]]
        Xv = [float(x.split(':')[1]) for x in line[1:]]
        return Xi, Xv, [y]

    def file_line_generator(self):
        print('train data files:', self.filenames)
        for fpath in self.filenames:
            f = file_io.FileIO(fpath, mode="r")
            line = f.readline()
            while line:
                line = line.strip()
                if line == '':
                    line = f.readline()
                    continue

                Xi, Xv, y = self.line2xy(line)
                yield Xi, Xv, y
                # yield dict({'feat_index': Xi, 'feat_value': Xv}), y
                line = f.readline()
            f.close()

    def create_dataset(self, batch_size, num_epochs=None):
        dataset = tf.data.Dataset.from_generator(self.file_line_generator,
                                                 (tf.int64, tf.float32, tf.int64),
                                                 (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([1]))
                                                 )
        dataset = dataset.map(lambda xi, xv, y: ({'feat_index': xi, 'feat_value': xv}, y))
        if num_epochs is not None and num_epochs >= 1:
            dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        return dataset

    def decode_csv_fn(self, line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.int32)
        labels = tf.reshape(labels, [-1])

        splits = tf.string_split(columns.values[1:], ':')

        id_vals = tf.reshape(splits.values, splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)

        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int64)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)

        feat_ids = tf.reshape(feat_ids, [-1])
        feat_vals = tf.reshape(feat_vals, [-1])

        #         indices = np.array(range(0,3))
        #         indices = indices.reshape(3,1)
        #         sparse_feature = tf.SparseTensor(indices, tf.reshape(feat_vals,[-1]), [3])
        #         dense_feature = tf.sparse.to_dense(sparse_feature)

        return {'feat_index': feat_ids, 'feat_value': feat_vals}, labels

    def create_dataset_from_file(self, batch_size, num_epochs=None):
        dataset_files = tf.data.Dataset.from_tensor_slices(self.filenames)
        dataset = dataset_files.interleave(
            lambda x:
                tf.data.TextLineDataset(x).map(self.decode_csv_fn,
                                               num_parallel_calls=8),  # 并行的进行map
            cycle_length=4,  # 控制并发读取数量
            block_length=1)

        # dataset_files = tf.data.Dataset.list_files(self.filenames)
        # dataset = dataset_files.apply(parallel_interleave(
        #     lambda x:
        #         tf.data.TextLineDataset(x).map(self.decode_csv_fn,
        #                                        num_parallel_calls=8),  # 并行的进行map
        #     cycle_length=4,  # 控制并发读取数量
        #     block_length=1))

        dataset = dataset.repeat(num_epochs)  # 重复次数
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)  # 实现流水线
        return dataset

    def generate_batch(self, batch_size, num_epochs=None):
        dataset = self.create_dataset(batch_size, num_epochs)

        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()

        return out_batch


def set_tfconfig_environ(choose_ps_as_evaluate=False):
    parse_argument()

    if "TF_CLUSTER_DEF" in os.environ:
        cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
        task_index = int(os.environ["TF_INDEX"])
        task_type = os.environ["TF_ROLE"]

        tf_config = dict()
        worker_num = len(cluster["worker"])
        FLAGS.worker_num = worker_num
        if task_type == "ps":
            # 把第一个ps转为evaluator
            if len(cluster["ps"]) >= 2 and choose_ps_as_evaluate:
                if task_index == 0:
                    tf_config["task"] = {"index": 0, "type": "evaluator"}
                    FLAGS.job_name = "evaluator"
                    FLAGS.task_index = 0
                else:
                    tf_config["task"] = {"index": task_index - 1, "type": task_type}
                    FLAGS.job_name = "ps"
                    FLAGS.task_index = task_index - 1
            else:
                tf_config["task"] = {"index": task_index, "type": task_type}
                FLAGS.job_name = "ps"
                FLAGS.task_index = task_index
        else:
            if task_index == 0:
                tf_config["task"] = {"index": 0, "type": "chief"}
            else:
                tf_config["task"] = {"index": task_index - 1, "type": task_type}
            FLAGS.job_name = "worker"
            FLAGS.task_index = task_index

        if worker_num == 1:
            cluster["chief"] = cluster["worker"]
            del cluster["worker"]
        else:
            cluster["chief"] = [cluster["worker"][0]]
            del cluster["worker"][0]

        if len(cluster["ps"]) >= 2 and choose_ps_as_evaluate:
            del cluster["ps"][0]

        tf_config["cluster"] = cluster
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))

    # if "INPUT_FILE_LIST" in os.environ:
    #     INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
    #     if INPUT_PATH:
    #         print("input path:", INPUT_PATH)
    #         FLAGS.train_data = INPUT_PATH.get(FLAGS.train_data)
    #         FLAGS.eval_data = INPUT_PATH.get(FLAGS.eval_data)
    #     else:  # for ps
    #         print("load input path failed.")
    #         FLAGS.train_data = None
    #         FLAGS.eval_data = None


def parse_argument():
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")

    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    os.environ["TF_ROLE"] = FLAGS.job_name
    os.environ["TF_INDEX"] = str(FLAGS.task_index)

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")

    cluster = {"worker": worker_spec, "ps": ps_spec}

    os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)


def main(_):
    print(FLAGS)
    # file_io.delete_recursively(FLAGS.checkpointDir)

    if len(FLAGS.ps_hosts) > 1 and len(FLAGS.worker_hosts) > 1:
        set_tfconfig_environ(FLAGS.choose_ps_as_evaluate)

    dfm_params = {
        'feature_size': 12026792,
        'field_size': 79,
        'k': 8,
        'use_fm': True,
        'use_deep': False,
        'dropout_keep_fm': [0.0, 0.0],
        'deep_layers': [32, 32, 1],
        'dropout_keep_deep': [0.5, 0.5, 0.5],
        'epoch': FLAGS.num_epochs,
        'batch_size': FLAGS.task_batch_size * FLAGS.worker_num,
        'learning_rate': FLAGS.learning_rate,
        'optimizer_type': 'adam',
        'verbose': 1,
        'random_seed': 1234,
        'loss_type': 'logloss',
        'eval_metric': 'auc',
        'l2': 0.01,
        'l2_fm': 0.01,
        'log_dir': FLAGS.checkpointDir,
        'bestModelPath': os.path.join(FLAGS.checkpointDir, 'keras.model'),
        'greater_is_better': True
    }

    feature_table = 'oss_ml_video_recommend_feature_mapping'
    feature_table_gen = oss_odps_table_generator(feature_table)

    max_feat_id = 0
    try:
        while True:
            line = feature_table_gen.next()
            col_arr = line.split(',')
            field_name = col_arr[0]
            feat_value = ','.join(col_arr[1:-2])
            feat_id = int(col_arr[-2])
            field_id = int(col_arr[-1])
            max_feat_id = feat_id if feat_id > max_feat_id else max_feat_id
    except StopIteration as si:
        print("feature_table_gen StopIteration")

    train_table = 'oss_ml_video_recommend_train_data_for_like'
    test_table = 'oss_ml_video_recommend_test_data_for_like'
    train_table_gen = oss_odps_table_generator(train_table)
    line = train_table_gen.next()
    field_size = len(line.split(' ')) - 1

    dfm_params['feature_size'] = max_feat_id + 1
    dfm_params['field_size'] = field_size

    print('feature_size: {}, field_size: {}'.format(dfm_params['feature_size'], dfm_params['field_size']))

    dfm = DeepFMEstimator(**dfm_params)
    # dfm.fit_on_libsvm(train_table, test_table)

    # devices = ['/device:CPU:0']
    # strategy = MirroredStrategy()
    # print(dfm.model.input_names)

    print(tf.VERSION)

    total = 2471795  # 16333644
    epoch_steps = total // dfm_params['batch_size']
    epochs_between_evals = 1
    save_checkpoints_steps = epoch_steps // 4
    max_steps = FLAGS.num_epochs * epoch_steps

    print('total: %d, epoch_steps: %d, save_checkpoints_steps: %d, max_steps: %d'
          % (total, epoch_steps, save_checkpoints_steps, max_steps))

    config = tf.estimator.RunConfig(model_dir=FLAGS.checkpointDir,
                                    tf_random_seed=123,
                                    save_summary_steps=save_checkpoints_steps,
                                    save_checkpoints_steps=save_checkpoints_steps,
                                    train_distribute=None)

    estimator = tf.estimator.Estimator(model_fn=dfm.deepfm_model_fn,
                                       model_dir=FLAGS.checkpointDir,
                                       config=config)

    train_input_fn = lambda: OssDataFeeder(train_table).create_dataset(dfm_params['batch_size'],
                                                                       num_epochs=FLAGS.num_epochs)
    test_input_fn = lambda: OssDataFeeder(test_table).create_dataset(dfm_params['batch_size'], num_epochs=None)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps)
    
    # 这里定义 exporters, 用于导出 bestModel
    serving_feature_spec = {
        "feat_index": parsing_ops.FixedLenFeature([dfm_params['field_size']], dtype=tf.int64),
        "feat_value": parsing_ops.FixedLenFeature([dfm_params['field_size']], dtype=tf.float32),
    }
    serving_input_receiver_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(
            serving_feature_spec))
    exporters = [
        BestExporter(
            name="best_exporter",
            serving_input_receiver_fn=serving_input_receiver_fn,
            exports_to_keep=5)
    ]
    eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, steps=None, throttle_secs=300, exporters=exporters)

    # for epoch in range(FLAGS.num_epochs):
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # for epoch in range(FLAGS.num_epochs):
    #     estimator.train(train_input_fn, max_steps=max_steps)
    #     results = estimator.evaluate(test_input_fn)
    #     for key in sorted(results):
    #         print('%s: %s' % (key, results[key]))

    # Evaluate accuracy.
    results = estimator.evaluate(input_fn=test_input_fn)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))

    print("after evaluate")

    # if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
    #     print("exporting model ...")
    #     serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    #     estimator.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
    # print("quit main")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--task_batch_size', type=int, default=64,
                        help='batch size for each task')

    parser.add_argument('--buckets', type=str,
                        default='./data/',
                        help='data directory')
    # parser.add_argument('--train_table', type=str, default='',
    #                     help='maxcompute table for train data')
    # parser.add_argument('--test_table', type=str, default='',
    #                     help='maxcompute table for test data')
    # parser.add_argument('--feature_table', type=str, default='',
    #                     help='maxcompute table for feature mapping data')

    parser.add_argument('--checkpointDir', type=str,
                        default='./checkpoint/',
                        help='checkpoint output directory')

    # 以下参数仅在多机多卡时有用
    parser.add_argument("--ps_hosts", type=str, default="", help="")
    parser.add_argument("--worker_hosts", type=str, default="", help="")
    parser.add_argument("--job_name", type=str, default="", help="One of 'ps', 'worker'")
    # Flags for defining the tf.train.Server
    parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
    parser.add_argument('--worker_num', type=int, default=1,
                        help='Number of train worker.')
    parser.add_argument('--save_checkpoints_steps', type=int, default=100,
                        help='Save checkpoints every this many steps')
    parser.add_argument('--choose_ps_as_evaluate', type=bool, default=False,
                        help='whether to choose one of ps server as evaluate')

    FLAGS, unparsed = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
