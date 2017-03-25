import tensorflow as tf
import time
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from utils import evaluation,make_records

class TrustSVD():
    def __init__(self,sess,args,
                 num_users,num_items,hidden_neuron,current_time,
                 R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,trust_matrix,
                 train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                 decay_epoch_step,lambda_value,
                 user_train_set, item_train_set, user_test_set, item_test_set,
                 result_path,date,data_name,
                 lambda_list,train_ratio,model_name):

        self.sess = sess
        self.args = args

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_neuron = hidden_neuron

        self.current_time = current_time

        self.R = R
        self.mask_R = mask_R
        self.C = C
        self.train_R = train_R
        self.train_mask_R = train_mask_R
        self.test_R = test_R
        self.test_mask_R = test_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings
        self.trust_matrix = tf.cast(trust_matrix , tf.float32)

        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.num_batch = int(self.num_users / float(self.batch_size)) + 1

        self.lr = tf.cast(lr,tf.float32)
        self.optimizer_method = optimizer_method
        self.display_step = display_step
        self.random_seed = random_seed


        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch

        self.lambda_value = lambda_value

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_acc_list = []
        self.test_avg_loglike_list = []

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set

        self.result_path = result_path
        self.date = date
        self.data_name = data_name

        self.lambda_value = lambda_list[0]
        self.lambda_t_value = lambda_list[1]

        self.train_ratio = train_ratio
        self.model_name = model_name

        self.earlystop_switch = False
        self.min_RMSE = 99999
        self.min_epoch = -99999
        self.patience = 0
        self.total_patience = 20

    def run(self):
        self.prepare_model()
        print ("========== End of Prepare the Model ==========")
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch_itr in range(self.train_epoch):
            if self.earlystop_switch:
                break
            else:
                self.train_model(epoch_itr)
                self.test_model(epoch_itr)

        make_records(self.result_path,self.test_acc_list,self.test_rmse_list,self.test_mae_list,self.test_avg_loglike_list,self.current_time,
                     self.args,self.model_name,self.data_name,self.train_ratio,self.hidden_neuron,self.random_seed,self.optimizer_method,self.lr)

    def prepare_model(self):
        ''''''
        ''' ================================== initialize the variable and constant ================================== '''
        self.input_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R")
        self.input_mask_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_mask_R")

        b_u = tf.get_variable(name="b_u", initializer=tf.zeros(shape=[self.num_users,1]),dtype=tf.float32)
        b_j = tf.get_variable(name="b_j", initializer=tf.zeros(shape=[self.num_items,1]),dtype=tf.float32)

        p_u = tf.get_variable(name="p_u", initializer=tf.truncated_normal(shape=[self.num_users, self.hidden_neuron],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        q_j = tf.get_variable(name="q_j", initializer=tf.truncated_normal(shape=[self.num_items, self.hidden_neuron],
                                         mean=0, stddev=0.03),dtype=tf.float32)

        y_i = tf.get_variable(name="y_i", initializer=tf.truncated_normal(shape=[self.num_items, self.hidden_neuron],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        w_v = tf.get_variable(name="w_v", initializer=tf.truncated_normal(shape=[self.num_users, self.hidden_neuron],
                                         mean=0, stddev=0.03),dtype=tf.float32)

        mu = tf.cast(np.sum(self.train_R) / float(self.num_train_ratings) , tf.float32)

        I_u = tf.reduce_sum(self.input_mask_R,1)
        U_j = tf.reduce_sum(self.input_mask_R,0)
        T_u = tf.reduce_sum(self.trust_matrix,1)
        T_v = tf.reduce_sum(self.trust_matrix,0)

        inverse_I_u = tf.div(tf.constant(1,tf.float32),I_u)
        inverse_U_j = tf.div(tf.constant(1,tf.float32), U_j)
        inverse_T_u = tf.div(tf.constant(1,tf.float32), T_u)
        inverse_T_v = tf.div(tf.constant(1,tf.float32), T_v)

        sqrt_inverse_I_u = tf.sqrt(tf.reshape(tf.where(tf.is_inf(inverse_I_u), tf.ones_like(inverse_I_u) * 0, inverse_I_u),[self.num_users,1]))
        sqrt_inverse_U_j = tf.sqrt(tf.reshape(tf.where(tf.is_inf(inverse_U_j), tf.ones_like(inverse_U_j) * 0, inverse_U_j),[self.num_items,1]))
        sqrt_inverse_T_u = tf.sqrt(tf.reshape(tf.where(tf.is_inf(inverse_T_u), tf.ones_like(inverse_T_u) * 0, inverse_T_u),[self.num_users,1]))
        sqrt_inverse_T_v = tf.sqrt(tf.reshape(tf.where(tf.is_inf(inverse_T_v), tf.ones_like(inverse_T_v) * 0, inverse_T_v),[self.num_users,1]))
        ''' ======================================================================================================== '''


        ''' ================================== make r_hat ================================== '''
        pre_r_hat1 = tf.matmul(b_u , tf.ones([1,self.num_items],dtype=tf.float32)) \
                     + tf.matmul(tf.ones([self.num_users,1],dtype=tf.float32) , tf.transpose(b_j))\
                     + mu * tf.ones([self.num_users,self.num_items],dtype=tf.float32)
        pre_r_hat2 = tf.matmul(p_u , tf.transpose(q_j))

        temp_r_hat3_1 = []
        temp_r_hat3_2 = []
        for user in range(self.num_users):
            user_specific_mask_r = self.input_mask_R[user,:]
            user_specific_trust_matrix = self.trust_matrix[user,:]
            zero = tf.constant(0, dtype=tf.float32)

            if I_u[user] == 0:
                temp_r_hat3_1.append(tf.zeros(shape=[self.hidden_neuron]))
            else:
                where = tf.not_equal(user_specific_mask_r, zero)
                indices = tf.cast(tf.where(where),tf.int32)
                indexed_y_i = tf.gather_nd(y_i, indices)
                sum_y_i = tf.reduce_sum(indexed_y_i, 0) * sqrt_inverse_I_u[user]
                temp_r_hat3_1.append(sum_y_i)

            # if np.sum(self.trust_matrix[user,:]) == 0:
            if T_u[user] == 0:
                temp_r_hat3_2.append(tf.zeros(shape=[self.hidden_neuron]))
            else:
                where = tf.not_equal(user_specific_trust_matrix, zero)
                indices = tf.cast(tf.where(where), tf.int32)
                indexed_w_v = tf.gather_nd(w_v, indices)
                sum_w_v = tf.reduce_sum(indexed_w_v, 0)  * sqrt_inverse_T_u[user]
                temp_r_hat3_2.append(sum_w_v)

        temp_r_hat3_1 = tf.stack(temp_r_hat3_1)
        temp_r_hat3_2 = tf.stack(temp_r_hat3_2)
        pre_r_hat3 = tf.matmul(temp_r_hat3_1 , tf.transpose(q_j)) + tf.matmul(temp_r_hat3_2 , tf.transpose(q_j))

        self.r_hat = pre_r_hat1 + pre_r_hat2 + pre_r_hat3
        ''' ======================================================================================================== '''


        ''' ================================== make t_hat ================================== '''
        self.t_hat = tf.matmul(p_u,tf.transpose(w_v))
        ''' ======================================================================================================== '''


        ''' ================================== make cost ================================== '''
        cost1 = 0.5 * tf.reduce_sum(tf.multiply(tf.square(self.r_hat - self.input_R) , self.input_mask_R)) \
                + 0.5 * self.lambda_t_value * tf.reduce_sum(tf.multiply(tf.square(self.t_hat - self.trust_matrix) , self.trust_matrix))

        cost2 = 0.5 * self.lambda_value * tf.matmul(tf.transpose(sqrt_inverse_I_u),tf.square(b_u)) \
                + 0.5 * self.lambda_value * tf.matmul(tf.transpose(sqrt_inverse_U_j),tf.square(b_j))

        pre_cost3 = tf.transpose(0.5 * self.lambda_value * sqrt_inverse_I_u + 0.5 * self.lambda_t_value * sqrt_inverse_T_u)
        frob_p_u = tf.reshape(tf.reduce_sum(tf.square(p_u),1),[self.num_users,1])
        cost3 = tf.matmul(pre_cost3 , frob_p_u)

        frob_q_j = tf.reshape(tf.reduce_sum(tf.square(q_j),1),[self.num_items,1])
        frob_y_i = tf.reshape(tf.reduce_sum(tf.square(y_i),1),[self.num_items,1])
        cost4 = 0.5 * self.lambda_value * tf.matmul(tf.transpose(sqrt_inverse_U_j),frob_q_j) \
                + 0.5 * self.lambda_value * tf.matmul(tf.transpose(sqrt_inverse_U_j),frob_y_i)

        frob_w_v = tf.reshape(tf.reduce_sum(tf.square(w_v),1),[self.num_users,1])
        cost5 = 0.5 * self.lambda_value * tf.matmul(tf.transpose(sqrt_inverse_T_v),frob_w_v)

        self.cost = tf.squeeze(cost1 + cost2 + cost3 + cost4 + cost5)
        #self.cost = tf.squeeze(cost1)# + cost3 + cost4 + cost5)
        ''' ======================================================================================================== '''

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr,0.9)
        else:
            raise ValueError("Optimizer Key ERROR")

        #self.optimizer = optimizer.minimize(self.cost)

        gvs = optimizer.compute_gradients(self.cost)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)


    def train_model(self,itr):
        start_time = time.time()

        _, Cost = self.sess.run(
            [self.optimizer, self.cost],
            feed_dict={self.input_R: self.train_R,
                       self.input_mask_R: self.train_mask_R})

        self.train_cost_list.append(Cost)

        if itr % self.display_step == 0:
            print ("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(Cost),
                   "Elapsed time : %d sec" % (time.time() - start_time))


    def test_model(self,itr):
        start_time = time.time()

        Cost,R_hat = self.sess.run(
            [self.cost,self.r_hat],
            feed_dict={self.input_R: self.test_R,
                       self.input_mask_R: self.test_mask_R})
        self.test_cost_list.append(Cost)

        Estimated_R = R_hat.clip(min=0, max=1)
        RMSE, MAE, ACC, AVG_loglikelihood = evaluation(self.test_R, self.test_mask_R, Estimated_R,
                                                       self.num_test_ratings)
        self.test_rmse_list.append(RMSE)
        self.test_mae_list.append(MAE)
        self.test_acc_list.append(ACC)
        self.test_avg_loglike_list.append(AVG_loglikelihood)

        if itr % self.display_step == 0:
            print("Testing //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(Cost),
                  "Elapsed time : %d sec" % (time.time() - start_time))
            print("RMSE = {:.4f}".format(RMSE), "MAE = {:.4f}".format(MAE), "ACC = {:.10f}".format(ACC),
                  "AVG Loglike = {:.4f}".format(AVG_loglikelihood))
            print("=" * 100)

        if RMSE <= self.min_RMSE:
            self.min_RMSE = RMSE
            self.min_epoch = itr
            self.patience = 0
        else:
            self.patience = self.patience + 1

        if (itr > 100) and (self.patience >= self.total_patience):
            self.test_rmse_list.append(self.test_rmse_list[self.min_epoch])
            self.test_mae_list.append(self.test_mae_list[self.min_epoch])
            self.test_acc_list.append(self.test_acc_list[self.min_epoch])
            self.test_avg_loglike_list.append(self.test_avg_loglike_list[self.min_epoch])
            self.earlystop_switch = True
            print ("========== Early Stopping at Epoch %d" %itr)

