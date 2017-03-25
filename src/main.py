from data_preprocessor import *
from TrustSVD import TrustSVD
from DAE import DAE
import tensorflow as tf
import time
import argparse

current_time = time.time()

parser = argparse.ArgumentParser(description='TrustSVD')
parser.add_argument('--model_name', choices=['TrustSVD'], default='TrustSVD')
parser.add_argument('--data_name', choices=['politic_old','politic_new'], default='politic_new')
parser.add_argument('--test_fold', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--train_epoch', type=int, default=2)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optimizer_method', choices=['Adam','Adadelta','Adagrad','RMSProp','GradientDescent','Momentum'],default='Adam')
parser.add_argument('--a', type=float, default=1)
parser.add_argument('--b', type=float, default=0)
parser.add_argument('--grad_clip', choices=['True', 'False'], default='True')  # True

parser.add_argument('--hidden_neuron', type=int, default=10)
parser.add_argument('--lambda_value', type=float, default=1000)
parser.add_argument('--lambda_t_value', type=float, default=0.001)

args = parser.parse_args()

random_seed = args.random_seed
tf.reset_default_graph()
np.random.seed(random_seed)
# np.random.RandomState
tf.set_random_seed(random_seed)

model_name = args.model_name
data_name = args.data_name
data_base_dir = "./data/"
path = data_base_dir + "%s" % data_name + "/"

if data_name == 'politic_new':
    num_users = 1537
    num_items = 7975
    num_total_ratings = 2999844
    num_voca = 13581
elif data_name == 'politic_old':
    num_users = 1540
    num_items = 7162
    num_total_ratings = 2779703
    num_voca = 10000
else:
    raise NotImplementedError("ERROR")

a = args.a
b = args.b

test_fold = args.test_fold
hidden_neuron = args.hidden_neuron

batch_size = 256
lr = args.lr
train_epoch = args.train_epoch
optimizer_method = args.optimizer_method
display_step = args.display_step
decay_epoch_step = 10000
decay_rate = 0.96
grad_clip = args.grad_clip

date = "0203"
result_path = './results/' + data_name + '/' + model_name + '/' + str(test_fold) +  '/' + str(current_time)+"/"

R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set \
    = read_rating(path, data_name,num_users, num_items,num_total_ratings, a, b, test_fold,random_seed)

X_dw = read_bill_term(path,data_name,num_items,num_voca)

print ("Type of Model : %s" %model_name)
print ("Type of Data : %s" %data_name)
print ("# of User : %d" %num_users)
print ("# of Item : %d" %num_items)
print ("Test Fold : %d" %test_fold)
print ("Random seed : %d" %random_seed)
print ("Hidden neuron : %d" %hidden_neuron)

with tf.Session() as sess:

    if model_name == "TrustSVD":
        lambda_value = args.lambda_value
        lambda_t_value = args.lambda_t_value
        lambda_list = [lambda_value, lambda_t_value]

        trust_matrix = read_trust(path, data_name, num_users)
        model = TrustSVD(sess,args,
                         num_users,num_items,hidden_neuron,current_time,
                         R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,trust_matrix,
                         train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                         decay_epoch_step,lambda_value,
                         user_train_set, item_train_set, user_test_set, item_test_set,
                         result_path,date,data_name,
                         lambda_list,test_fold,model_name)
    else:
        raise NotImplementedError("ERROR")

    model.run()


