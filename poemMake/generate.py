#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from vocab import *
from rhyme import RhymeDict
from word2vec import get_word_embedding
from data_utils import *
from collections import deque
import tensorflow as tf
from tensorflow.contrib import rnn

flags = tf.app.flags   # tf定义了tf.app.flags，用于支持接受命令行传递参数，第一个是参数名称，第二个参数是默认值，第三个是参数描述
#定义常量
# flags.DEFINE_string("data_dir", "/tmp/mnist-data",
#                     "Directory for storing mnist data")
# #只下载数据，不做处理
# flags.DEFINE_boolean("download_only", False,
#                      "Only perform downloading of data; Do not proceed to "
#                      "session preparation, model definition or training")
# 第 1 步：命令行参数解析，获取集群的信息 ps_hosts 和 worker_hosts，
# 以及当前节点的角色信息 job_name 和 task_index。
# index 从 0 开始。 0 代表用来初始化变量的第一个任务
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
#每台机器的gpu个数，如果不使用GPU，设置为0
flags.DEFINE_integer("num_gpus", 0, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
# 在同步训练模式下，设置收集的工作节点的数量。默认就是工作节点的总数
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
# flags.DEFINE_integer("hidden_units", 100,
#                      "Number of units in the hidden layer of the NN")
# flags.DEFINE_integer("train_steps", 200,
#                      "Number of (global) training steps to perform")
# flags.DEFINE_integer("batch_size", 100, "Training batch size")
# flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
#使用同步训练或者异步训练
flags.DEFINE_boolean(
    "sync_replicas", False,
    "Use the sync_replicas (synchronized replicas) mode, "
    "wherein the parameter updates from workers are aggregated "
    "before applied to avoid stale gradients")
# 如果服务器已经存在，采用 gRPC 协议通信；如果不存在，采用进程间通信
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
# 参数服务器主机
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
# 工作节点主机
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
# 本作业是工作节点还是参数服务器
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 通过os.environ[key]=value来设置环境变量

_model_path = os.path.join(save_dir, 'model')

_NUM_UNITS = 128  # 隐藏层节点数目
_NUM_LAYERS = 4  # 隐藏层层数
_BATCH_SIZE = 64  # 一次批量处理的数据记录数

# 将神经网络模型的构建、训练和测试放在一个类中不是很好的做法
class Generator:

    def __init__(self):  # 构建encoder decoder模型，时刻应该考虑到batch_size的大小，因为训练的时候是以batch_size为单位的。

        if FLAGS.job_name is None or FLAGS.job_name == "":
            raise ValueError("Must specify an explicit `job_name`")
        if FLAGS.task_index is None or FLAGS.task_index == "":
            raise ValueError("Must specify an explicit `task_index`")

        print("job name = %s" % FLAGS.job_name)
        print("task index = %d" % FLAGS.task_index)

        # Construct the cluster and start the server 读取集群的描述信息
        ps_spec = FLAGS.ps_hosts.split(",")
        worker_spec = FLAGS.worker_hosts.split(",")

        # Get the number of workers.
        num_workers = len(worker_spec)
        # 创建 TensorFlow 集群描述对象
        # 第 2 步：创建当前任务节点的服务器
        cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
        server = tf.train.Server(  # 每个Task对应一个tf.train.Server实例
                cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
            # 如果是参数服务器，直接启动即可。这时,进程就会阻塞在这里
            # 下面的 tf.train.replica_device_setter 代码会将参数指定给 ps_server 保管

            # 第 3 步：如果当前节点是参数服务器，则调用 server.join()无休止等待；如果是工作节点，则执行第 4 步
        if FLAGS.job_name == "ps":
            server.join()
        is_chief= (FLAGS.task_index == 0)
        # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
        with tf.device(tf.train.replica_device_setter(
                cluster=cluster
        )):
        embedding = tf.Variable(tf.constant(0.0, shape=[VOCAB_SIZE, _NUM_UNITS]), trainable = False)
        self._embed_ph = tf.placeholder(tf.float32, [VOCAB_SIZE, _NUM_UNITS])  #准备接收字库中的词向量矩阵
        self._embed_init = embedding.assign(self._embed_ph)

        self.encoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(_NUM_UNITS)] * _NUM_LAYERS)
        self.encoder_init_state = self.encoder_cell.zero_state(_BATCH_SIZE, dtype = tf.float32)  # 实际init_state的shape是[_BATCH_SIZE,_NUM_UNITS]
        self.encoder_inputs = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.encoder_lengths = tf.placeholder(tf.int32, [_BATCH_SIZE])  # RNN一次处理的序列长度
        _, self.encoder_final_state = tf.nn.dynamic_rnn(
                cell = self.encoder_cell,
                initial_state = self.encoder_init_state,
                inputs = tf.nn.embedding_lookup(embedding, self.encoder_inputs),  #难点在于理解embedding_lookup()
                sequence_length = self.encoder_lengths,
                scope = 'encoder')  # encoder_final_state指的是encoder最后一个时刻的中间隐藏层状态

        self.decoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(_NUM_UNITS)] * _NUM_LAYERS)
        self.decoder_init_state = self.encoder_cell.zero_state(_BATCH_SIZE, dtype = tf.float32)  # tf1.2之前的版本不适用该种形式
        self.decoder_inputs = tf.placeholder(tf.int32, [_BATCH_SIZE, None])  # decoder的输入有很多种模式
        self.decoder_lengths = tf.placeholder(tf.int32, [_BATCH_SIZE])
        outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                cell = self.decoder_cell,
                initial_state = self.decoder_init_state,
                inputs = tf.nn.embedding_lookup(embedding, self.decoder_inputs),
                sequence_length = self.decoder_lengths,
                scope = 'decoder')  # outputs指的是decoder每一时刻的输出（实际不是输出值）, self.decoder_final_state同上

        with tf.variable_scope('decoder'):  # 共享变量，是RNN最后一个隐藏层与输出层相接的参数而不是输入层与第一层隐藏层相接的参数
            softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS, VOCAB_SIZE])
            softmax_b = tf.get_variable('softmax_b', [VOCAB_SIZE])

        logits = tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, _NUM_UNITS]), softmax_w),
                bias = softmax_b)
        self.probs = tf.nn.softmax(logits)  # 对上面的outputs变换为真正的输出，是一种概率

        self.targets = tf.placeholder(tf.int32, [_BATCH_SIZE, None])  # 实际目标值
        labels = tf.one_hot(tf.reshape(self.targets, [-1]), depth = VOCAB_SIZE)
        loss = tf.nn.softmax_cross_entropy_with_logits(
                logits = logits,
                labels = labels)
        self.loss = tf.reduce_mean(loss)  # 整体的损失函数

        self.learn_rate = tf.Variable(0.0, trainable = False)
        self.opt_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)

        if FLAGS.sync_replicas:
            if FLAGS.replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = FLAGS.replicas_to_aggregate

            # 使用 SyncReplicasOptimizer 作为优化器，并且是在图间复制情况下
            # 在图本复制情况下将所有的梯度平均就可以了
            opt = tf.train.SyncReplicasOptimizer(  # 创建同步训练优化器，对原有优化器扩展
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                name="generate_sync_replicas")
        if FLAGS.sync_replicas:
        local_init_op = opt.local_step_init_op
        if is_chief:
            # 所有的进行计算的工作节点里的一个主工作节点（chief）
            # 这个主节点负责初始化参数、模型的保存、概要的保存等
            local_init_op = opt.chief_init_op

        ready_for_local_init_op = opt.ready_for_local_init_op

        # 同步训练模式所需的初始令牌和主队列
        # Initial token and chief queue runners required by the sync_replicas mode
        chief_queue_runner = opt.get_chief_queue_runner()
        sync_init_op = opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        if FLAGS.sync_replicas:
            # 创建一个监管程序，用于统计训练模型过程中的信息
            # logdir 是保存和加载模型的路径
            # 启动就会去这个 logdir 目录看是否有检查点文件，有的话就本动加载
            # 没有就用 init_op 指定的初始化参数
            # 主工作节点（chief）负责模型参数初始化等工作
            # 在这个过程中，其他工作节点等待主节点完成初始化工作，初始化完成后，一起开始训练数据
            # global_step 的值是所有计算节点共享的
            # 在执行损失函数最小值的时候会本动加 1，通过 global_step 能知道所有计算节点一共计算了多少步
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=save_dir,
                init_op=init_op,
                local_init_op=local_init_op,
                ready_for_local_init_op=ready_for_local_init_op,
                recovery_wait_secs=1)
        else:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=save_dir,
                init_op=init_op,
                recovery_wait_secs=1)
        self.saver = tf.train.Saver(tf.global_variables())
        self.int2ch, self.ch2int = get_vocab()  # 需要根本的词库

    def _init_vars(self, sess):
        ckpt = tf.train.get_checkpoint_state(save_dir)  # 检查点文件
        if not ckpt or not ckpt.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            sess.run(init_op)
            sess.run([self._embed_init], feed_dict = {
                self._embed_ph: get_word_embedding(_NUM_UNITS)})  # 将由训练诗句生成的词向量矩阵传入到_embed_ph占位符,这里规定的词向量维度就是隐藏层单元数了
        else:
            self.saver.restore(sess, ckpt.model_checkpoint_path)  #有恢复意味着后面一定有保存。

    def _train_a_batch(self, sess, kw_mats, kw_lens, s_mats, s_lens):  # 以batch_size=1的形式训练
        total_loss = 0
        for idx in range(4):
            encoder_feed_dict = {self.encoder_inputs: kw_mats[idx],
                    self.encoder_lengths: kw_lens[idx]}
            if idx > 0:
                encoder_feed_dict[self.encoder_init_state] = state
            state = sess.run(self.encoder_final_state,
                    feed_dict = encoder_feed_dict)  # feed_dict一定是placeholder类型量，而返回的则是某些想要得到的计算量
            state, loss, _ = sess.run([self.decoder_final_state, self.loss, self.opt_op], feed_dict = {
                self.decoder_init_state: state,
                self.decoder_inputs: s_mats[idx][:,:-1],  # decoder的inputs不要包含<EOS>，包含<GO>
                self.decoder_lengths: s_lens[idx],
                self.targets: s_mats[idx][:,1:]})  # decoder的target不包含<GO>,但是包含<EOS>
            total_loss += loss
        print "loss = %f" %(total_loss/4)

    def train(self, n_epochs = 6, learn_rate = 0.002, decay_rate = 0.97):  # 训练

        is_chief = (FLAGS.task_index == 0)
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps",
                            "/job:worker/task:%d" % FLAGS.task_index])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
        # 主工作节点（chief），即 task_index 为 0 的节点将会初始化会话
        # 其余的工作节点会等待会话被初始化后进行计算
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        if FLAGS.existing_servers:
            server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
            print("Using existing server at: %s" % server_grpc_url)

            # 创建TensorFlow 会话对象，用于执行TensorFlow 图计算
            # prepare_or_wait_for_session 需要参数初始化完成且主节点也准备好后，才开始训练
            sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
        else:
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)

        if FLAGS.sync_replicas and is_chief:
            # Chief worker will start the chief queue runner and call the init op.
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        # Perform training 执行分布式模型训练
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)

        print("Start training RNN enc-dec model ...")  #这是训练起点
        with tf.Session() as sess:
            self._init_vars(sess)
            try:
                for epoch in range(n_epochs):  # 迭代次数
                    batch_no = 0
                    sess.run(tf.assign(self.learn_rate, learn_rate * decay_rate ** epoch))
                    for kw_mats, kw_lens, s_mats, s_lens in batch_train_data(_BATCH_SIZE):  # 批处理方式，返回的四个数字矩阵，代表其位置
                        print "[Training Seq2Seq] epoch = %d/%d, line %d to %d ..." \
                                %(epoch, n_epochs, batch_no*_BATCH_SIZE, (batch_no+1)*_BATCH_SIZE),
                        self._train_a_batch(sess, kw_mats, kw_lens, s_mats, s_lens)
                        batch_no += 1
                        if 0 == batch_no%32:
                            self.saver.save(sess, _model_path)
                            print "[Training Seq2Seq] The temporary model has been saved."
                    self.saver.save(sess, _model_path)
                print "Training has finished."
            except KeyboardInterrupt:
                print "\nTraining is interrupted."

    def generate(self, keywords):  # 通过关键字生成诗句，相当于模型的预测，需要将encoder decoder串接起来。
        sentences = []
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if not ckpt or not ckpt.model_checkpoint_path:
            self.train(1)
        with tf.Session() as sess:
            self._init_vars(sess)
            rdict = RhymeDict()
            length = -1
            rhyme_ch = None
            for idx, keyword in enumerate(keywords):
                kw_mat = fill_np_matrix([[self.ch2int[ch] for ch in keyword]], _BATCH_SIZE, VOCAB_SIZE-1)
                kw_len = fill_np_array([len(keyword)], _BATCH_SIZE, 0)  # 因为是测试阶段，所以kw_mat和kw_len需要在此处生成，而不是像训练阶段通过batch_train_data生成
                encoder_feed_dict = {self.encoder_inputs: kw_mat,  # 这里的kw_mat是需要做成word embedding，
                # 所以即使字库中所有的词向量都已经在模型运行之前生成但是仍然需要embedding
                        self.encoder_lengths: kw_len}
                if idx > 0:
                    encoder_feed_dict[self.encoder_init_state] = state
                state = sess.run(self.encoder_final_state,
                        feed_dict = encoder_feed_dict)
                sentence = u''
                decoder_inputs = np.zeros([_BATCH_SIZE, 1], dtype = np.int32)
                decoder_lengths = fill_np_array([1], _BATCH_SIZE, 0)
                i = 0
                while True:
                    probs, state = sess.run([self.probs, self.decoder_final_state], feed_dict = {
                        self.decoder_init_state: state,
                        self.decoder_inputs: decoder_inputs,
                        self.decoder_lengths: decoder_lengths})
                    # 从此之后都是对decoder输出结果props的操作，一个判断其押韵效果，另一个是将每一个decoder输出向量依照字库int2ch转变为汉字
                    prob_list = probs.tolist()[0]
                    prob_list[0] = 0.
                    if length > 0:
                        if i  == length:
                            prob_list = [.0]*VOCAB_SIZE
                            prob_list[-1] = 1.
                        elif i == length-1:
                            for j, ch in enumerate(self.int2ch):
                                if  0 == j or VOCAB_SIZE-1 == j:
                                    prob_list[j] = 0.
                                else:
                                    rhyme = rdict.get_rhyme(ch)
                                    tone = rdict.get_tone(ch)
                                    if (1 == idx and 'p' != tone) or \
                                            (2 == idx and (rdict.get_rhyme(rhyme_ch) == rhyme or 'z' != tone)) or \
                                            (3 == idx and (ch == rhyme_ch or rdict.get_rhyme(rhyme_ch) != rhyme or 'p' != tone)):
                                        prob_list[j] = 0.
                        else:
                            prob_list[-1] = 0.
                    else:
                        if i != 5 and i != 7:
                            prob_list[-1] = 0.
                    prob_sums = np.cumsum(prob_list)
                    if prob_sums[-1] == 0.:
                        prob_list = probs.tolist()[0]
                        prob_sums = np.cumsum(prob_list)
                    for j in range(VOCAB_SIZE-1, -1, -1):
                        if random.random() < prob_list[j]/prob_sums[j]:
                            ch = self.int2ch[j]
                            break
                    #ch = self.int2ch[np.argmax(prob_list)]
                    if idx == 1 and i == length-1:
                        rhyme_ch = ch
                    if ch == self.int2ch[-1]:
                        length = i
                        break
                    else:
                        sentence += ch
                        decoder_inputs[0,0] = self.ch2int[ch]
                        i += 1
                #uprintln(sentence)
                sentences.append(sentence)
        return sentences


if __name__ == '__main__':
    generator = Generator()
    kw_train_data = get_kw_train_data()
    for row in kw_train_data[100:]:
        uprintln(row)
        generator.generate(row)
        print

