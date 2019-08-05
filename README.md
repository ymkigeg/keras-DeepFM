# keras-DeepFM  
Implementation of DeepFM using keras. And train on libsvm format file  
  
  
deepfm_esitmator.py  
自定义estimator来实现分布式的训练，也可以单机，成功  
  
tf_keras_estimator_deepFM.py  
先定义tf.keras.Model，然后调用 tf.keras.estimator.model_to_estimator 转为 estimator，失败  
  
报错如下：  
```
  File "/usr/lib/python2.7/site-packages/tensorflow/python/training/session_manager.py", line 519, in _try_run_local_init_op
    is_ready_for_local_init, msg = self._model_ready_for_local_init(sess)
  File "/usr/lib/python2.7/site-packages/tensorflow/python/training/session_manager.py", line 504, in _model_ready_for_local_init
    "Model not ready for local init")
  File "/usr/lib/python2.7/site-packages/tensorflow/python/training/session_manager.py", line 548, in _ready
    ready_value = sess.run(op)
  File "/usr/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1020, in run
    run_metadata_ptr)
  File "/usr/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1258, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1439, in _do_run
    run_metadata)
  File "/usr/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1458, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Cannot colocate nodes 'training/Adam/gradients/feature_embeddings/GatherV2_grad/Shape' and 'training/Adam/gradients/concat_3/axis: Cannot merge devices with incompatible jobs: '/job:worker/task:0' and '/job:ps/task:0'
	 [[Node: training/Adam/gradients/feature_embeddings/GatherV2_grad/Shape = Const[_class=["loc:@feature_embeddings/GatherV2", "loc:@feature_embeddings/embeddings"], dtype=DT_INT64, value=Tensor<type: int64 shape: [2] values: 329109 8>, _device="/job:worker/task:0"]()]]
```
