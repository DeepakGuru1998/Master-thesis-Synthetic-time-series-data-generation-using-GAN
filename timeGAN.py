# Import necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import GradientTape, sqrt, ones_like, zeros_like, reduce_mean, convert_to_tensor, float32
from tensorflow.nn import moments
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, LeakyReLU, BatchNormalization, ReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, KLDivergence
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras import backend
import joblib, os, random
from subprocess import PIPE, Popen
import time

def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)

# Create model
class TimeGAN:
    def __init__(self, gen_dim, disc_dim, emb_dim, rec_dim, sup_dim, num_features=8, num_latent=4, lr=0.0005, eta=10, gamma=1):
        self.gen_dim = gen_dim
        self.disc_dim = disc_dim
        self.emb_dim = emb_dim
        self.rec_dim = rec_dim
        self.sup_dim = sup_dim
        self.num_features = num_features
        self.num_latent = num_latent
        self.lr = lr
        self.eta = eta
        self.gamma = gamma
    
        # -----------------
        # Build neural nets
        # -----------------
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.embedder = self.build_embedder()
        self.recovery = self.build_recovery()
        self.supervisor = self.build_supervisor()
    
        # Define input
        real = Input(shape=(None, self.num_features))
        noise = Input(shape=(None, self.num_features))
        
        # -----------------------------
        # Building the Autoencoder part
        # -----------------------------
        emb_seq = self.embedder(real)
        rec_seq = self.recovery(emb_seq)
        
        self.autoencoder = Model(real, rec_seq, name='Autoencoder')
        
        # ------------------------------------
        # Building Adversarial Supervisor part
        # ------------------------------------
        gen_seq = self.generator(noise)
        sup_seq = self.supervisor(gen_seq)
        Y_fake = self.discriminator(sup_seq)
        
        self.adv_sup = Model(noise, Y_fake, name='Adversarial Supervised')
        
        # -------------------------------------
        # Building Generator-Recovery connection
        # -------------------------------------
        rec_seq_gen = self.recovery(sup_seq)
        
        self.rec_gen = Model(noise, rec_seq_gen, name='Generator-Recovery Connection')
        
        # -------------------------------------------
        # Building Embedding-Discriminator connection
        # -------------------------------------------
        Y_real = self.discriminator(emb_seq)
        
        self.disc_emb = Model(real, Y_real, name='Embedder-Discriminator Connection')
        
        # -----------------------------
        # Building the regular GAN part
        # -----------------------------        
        Y_fake_e = self.discriminator(gen_seq)
        
        self.adversarial = Model(noise, Y_fake_e, name='Adversarial')
        
        self._mse = MeanSquaredError()
        self._bce = BinaryCrossentropy()
        self._kld = KLDivergence()
        
    def build_generator(self):
        model = Sequential()
        model.add(LSTM(self.gen_dim[0], input_shape=(None, self.num_features), return_sequences=True))
        model.add(LSTM(self.gen_dim[1], return_sequences=True))
        model.add(LSTM(self.gen_dim[2], return_sequences=True))
        model.add(Dense(self.num_latent, activation='sigmoid'))
        
        seq = Input(shape=(None, self.num_features))
        x = model(seq)
        
        return Model(seq, x, name='Generator')
    
    def build_discriminator(self):
        #Many to One Discriminator since it gives only one output at the end (True or False)
        model = Sequential()
        model.add(LSTM(self.disc_dim[0], input_shape=(None, self.num_latent), return_sequences=True))
        model.add(LSTM(self.disc_dim[1], return_sequences=True))
        model.add(LSTM(self.disc_dim[2], return_sequences=True))
        model.add(Dense(1, activation='sigmoid'))

        z = Input(shape=(None, self.num_latent))
        validity = model(z)

        return Model(z, validity, name='Discriminator')
    
    def build_embedder(self):
        
        model = Sequential()
        model.add(LSTM(self.emb_dim[0], input_shape=(None, self.num_features), return_sequences=True))
        model.add(LSTM(self.emb_dim[1], return_sequences=True))
        model.add(LSTM(self.emb_dim[2], return_sequences=True))
        model.add(Dense(self.num_latent, activation='sigmoid'))
        
        seq = Input(shape=(None, self.num_features))
        x = model(seq)
        
        return Model(seq, x, name='Embedder')
    
    def build_recovery(self):
        
        model = Sequential()
        model.add(LSTM(self.rec_dim[0], input_shape=(None, self.num_latent), return_sequences=True))
        model.add(LSTM(self.rec_dim[1], return_sequences=True))
        model.add(LSTM(self.rec_dim[2], return_sequences=True))
        model.add(Dense(self.num_features, activation='sigmoid'))
        
        seq = Input(shape=(None, self.num_latent))
        x = model(seq)
        
        return Model(seq, x, name='Recovery')
    
    def build_supervisor(self):
        
        model = Sequential()
        model.add(LSTM(self.sup_dim[0], input_shape=(None, self.num_latent), return_sequences=True))
        model.add(LSTM(self.sup_dim[1], return_sequences=True))
        model.add(Dense(self.num_latent, activation='sigmoid'))
        
        seq = Input(shape=(None, self.num_latent))
        x = model(seq)
        
        return Model(seq, x, name='Supervisor')
    
    def train_autoencoder(self, real, opt):
        with GradientTape() as tape:
            rec = self.autoencoder(real)
            embedding_loss_t0 = self._mse(real, rec)
            e_loss_0 = 10*tf.sqrt(embedding_loss_t0)
        
        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        
        return tf.sqrt(embedding_loss_t0)
    
    def train_supervisor(self, real, opt):
        with GradientTape() as tape:
            emb_seq = self.embedder(real)
            sup_seq = self.supervisor(emb_seq)
            g_loss_s = self._mse(emb_seq[:, 1:, :], sup_seq[:, :-1, :])

        var_list = self.supervisor.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        
        return g_loss_s
    
    def train_embedder(self, real, opt):
        with GradientTape() as tape:
            emb_seq = self.embedder(real)
            sup_seq = self.supervisor(emb_seq)
            g_loss_s = self._mse(emb_seq[:, 1:, :], sup_seq[:, :-1, :])

            rec = self.autoencoder(real)
            embedding_loss_t0 = self._mse(real, rec)
            e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * g_loss_s

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        
        return tf.sqrt(embedding_loss_t0)
            
    def train_generator(self, real, noise, opt):
        with GradientTape() as tape:
            # Unsupervised generator loss
            y_fake = self.adv_sup(noise)
            g_loss_u = self._bce(y_true=np.ones_like(y_fake), y_pred=y_fake)

            y_fake_e = self.adversarial(noise)
            g_loss_u_e = self._bce(y_true=np.ones_like(y_fake_e), y_pred=y_fake_e)
            
            # Supervised generator loss
            emb_seq = self.embedder(real)
            sup_seq = self.supervisor(emb_seq)
            g_loss_s = self._mse(emb_seq[:, 1:, :], sup_seq[:, :-1, :])

            # Moment generator loss
            rec_seq = self.rec_gen(noise)
            x_d = tf.convert_to_tensor(real, dtype='float32')
            rec = tf.convert_to_tensor(rec_seq, dtype='float32')
            y_true_mean, y_true_var = moments(x=x_d, axes=[0])
            y_pred_mean, y_pred_var = moments(x=rec, axes=[0])
            g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
            g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
            g_loss_m = g_loss_mean + g_loss_var

            g_loss = (g_loss_u + self.gamma * g_loss_u_e + self.eta * tf.sqrt(g_loss_s) + 100* g_loss_m)

        var_list = self.generator.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(g_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        
        return g_loss_u, g_loss_s, g_loss_m, g_loss
    
    def train_discriminator(self, real, noise, opt):
        with GradientTape() as tape:
            # Loss from real samples
            y_real = self.disc_emb(real)
            d_loss_real = self._bce(y_true=np.ones_like(y_real), y_pred=y_real)
            
            # Loss from fake samples via Supervisor
            y_fake = self.adv_sup(noise)
            d_loss_fake = self._bce(y_true=np.zeros_like(y_fake), y_pred=y_fake)
            
            # Loss unsupervised - regular GAN
            y_fake_e = self.adversarial(noise)
            d_loss_fake_e = self._bce(y_true=np.zeros_like(y_fake_e), y_pred=y_fake_e)
            
            d_loss = d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e
        
        if d_loss > 0.15:
            var_list = self.discriminator.trainable_variables
            gradients = tape.gradient(d_loss, var_list)
            opt.apply_gradients(zip(gradients, var_list))
        return d_loss, d_loss_real, d_loss_fake, d_loss_fake_e
    
    def w_loss(self, y_true, y_pred):
        return backend.mean(y_true*y_pred)

    def get_samples(self, data_set, num_samples=30):
        
#       seq_length = random.randrange(30,601,30)  #variable sequence length
        seq_length = 60
           
        temp_data=[]
        for j in data_set:
            ran= [random.randrange(1, len(j)-seq_length, 1) for i in range(num_samples)]  #30 samples of data generated (total 120 for all 4 data sets combined)
            for k in ran:
                _x = j[k:k + seq_length]
                temp_data.append(_x)
        train_data = np.array(temp_data)

        return train_data
    
    def train(self, data_set, epochs=50, pre_epochs=200, plot_loss=False):      
        # --------------------
        # Autoencoder training
        # --------------------
        print('Starting Autoencoder pre-training', flush=True)
        
        emb_loss_log = []
        autoencoder_opt = Adam(learning_rate=self.lr)
        for epoch in range(pre_epochs):
            train_data = self.get_samples(data_set)
            
            auto_loss = self.train_autoencoder(train_data, autoencoder_opt)
            emb_loss_log.append(auto_loss.numpy())
            
        # -------------------
        # Supervisor training
        # -------------------
        print('Starting Supervisor pre-training', flush=True)
        
        sup_loss_log = []
        supervisor_opt = Adam(learning_rate=self.lr)
        for epoch in range(pre_epochs):
            
            train_data = self.get_samples(data_set)

            sup_loss = self.train_supervisor(train_data, supervisor_opt)
            sup_loss_log.append(sup_loss.numpy())
            
        print('--------------------------------', flush=True)
        # --------------
        # Joint training
        # --------------
        print('Starting Joint training', flush=True)
        
        generator_opt = Adam(learning_rate=self.lr)
        embedder_opt = Adam(learning_rate=self.lr)
        discriminator_opt = Adam(learning_rate=self.lr)

        g_u_loss_log = []
        g_s_loss_log = []
        g_v_loss_log = []
        g_tot_loss_log = []
        e_loss_log = []
        d_loss_real_log = []
        d_loss_fake_log = []
        d_loss_fake_e_log = []
        d_tot_loss_log = []
        
        for epoch in range(1, epochs+1):
            gl = 0
            gl_s = 0
            gl_u = 0
            gl_v = 0
            el = 0
            
            for _ in range(2):
                train_data = self.get_samples(data_set)
                num_samples = train_data.shape[0]
                batch_size = train_data.shape[1] 
                    
                noise = np.random.uniform(-1, 1, train_data.shape)

                # ---------------
                # Train Generator
                # ---------------
                g_loss_u, g_loss_s, g_loss_v, g_loss = self.train_generator(train_data, noise, generator_opt)
                gl += (g_loss_u+self.eta*tf.sqrt(g_loss_s)+100*g_loss_v).numpy()
                gl_s += self.eta*g_loss_s.numpy()**0.5
                gl_u += g_loss_u.numpy()
                gl_v += 100*g_loss_v.numpy()

                # --------------
                # Train Embedder
                #---------------
                e_loss_0 = self.train_embedder(train_data, embedder_opt)
                el += e_loss_0.numpy()
            
            g_u_loss_log.append(gl_u/2)
            g_s_loss_log.append(gl_s/2)
            g_v_loss_log.append(gl_v/2)
            g_tot_loss_log.append(gl/2)
            e_loss_log.append(el/2)

            # -------------------
            # Train Discriminator
            # -------------------       
            d_loss, d_loss_real, d_loss_fake, d_loss_fake_e = self.train_discriminator(train_data, noise, discriminator_opt)
            
            if d_loss.numpy() > 0.15:
                d_loss_real_log.append(d_loss_real.numpy())
                d_loss_fake_log.append(d_loss_fake.numpy())
                d_loss_fake_e_log.append(d_loss_fake_e.numpy())
                d_tot_loss_log.append(d_loss.numpy())
            else:
                d_loss_real_log.append(0)
                d_loss_fake_log.append(0)
                d_loss_fake_e_log.append(0)
                d_tot_loss_log.append(0)
            
            loss_dict = {'pre_embedder_loss': emb_loss_log,
                         'pre_supervisor_loss': sup_loss_log,
                         'gen_unsupervised_loss': g_u_loss_log, 
                         'gen_supervised_loss': g_s_loss_log, 
                         'gen_moment_loss': g_v_loss_log, 
                         'total_gen_loss': g_tot_loss_log, 
                         'disc_real_loss': d_loss_real_log, 
                         'disc_fake_loss': d_loss_fake_log, 
                         'disc_fake_e_loss': d_loss_fake_e_log, 
                         'total_disc_loss': d_tot_loss_log, 
                         'embedder_loss': e_loss_log}
            # Print the progress
            if epoch % 10 == 0:
                print ("%d [D loss: %f, D Real loss: %f, D Fake loss: %f] [G loss: %f, G loss u: %f, G loss s: %f, G loss v: %f]" \
                   % (epoch, d_loss, d_loss_real, d_loss_fake, gl/2, gl_u/2, gl_s/2, gl_v/2), flush=True) 
                
            if epoch % 1000 == 0 or epoch == epochs+1:
                m_name = 'tGAN'+str(epoch)+'.pkl'
                l_name = 'loss_dict'+str(epoch)+'.pkl'
                self.save_model(m_name)
                joblib.dump(loss_dict, l_name)
        
                def copyFile2Hdfs(filename): 
                    ## **** mentions path
                    hdfs_filepath = '****'+filename
                    command = Popen(['hdfs', 'dfs', '-test', '-e', hdfs_filepath], stdout=PIPE, stderr=PIPE)
                    s_output, s_err = command.communicate()
                    s_return = command.returncode
                    if not(s_return): #if file exists
                        command = Popen(['hdfs', 'dfs', '-rm', hdfs_filepath], stdout=PIPE, stderr=PIPE)
                        command.communicate()
                    command = Popen(["hdfs", "dfs", "-put", filename, hdfs_filepath], stdin=PIPE, bufsize=-1)
                    command.communicate()
                
                copyFile2Hdfs(m_name)
                copyFile2Hdfs(l_name)
            
        all_loss_dict = {'pre_embedder_loss': emb_loss_log,
                         'pre_supervisor_loss': sup_loss_log,
                         'gen_unsupervised_loss': g_u_loss_log, 
                         'gen_supervised_loss': g_s_loss_log, 
                         'gen_moment_loss': g_v_loss_log, 
                         'total_gen_loss': g_tot_loss_log, 
                         'disc_real_loss': d_loss_real_log, 
                         'disc_fake_loss': d_loss_fake_log, 
                         'disc_fake_e_loss': d_loss_fake_e_log, 
                         'total_disc_loss': d_tot_loss_log, 
                         'embedder_loss': e_loss_log}
        
        return all_loss_dict
    
    def save_model(self, path):
        def unpack(model, training_config, weights):
            restored_model = deserialize(model)
            if training_config is not None:
                restored_model.compile(**saving_utils.compile_args_from_training_config(training_config))
            restored_model.set_weights(weights)
            return restored_model
        
        def make_keras_picklable():
            def __reduce__(self):
                model_metadata = saving_utils.model_metadata(self)
                training_config = model_metadata.get("training_config", None)
                model = serialize(self)
                weights = self.get_weights()
                return (unpack, (model, training_config, weights))
            cls = Model
            cls.__reduce__=__reduce__    
        make_keras_picklable()
        try:
            joblib.dump(self, path)
        except:
            raise Exception('Please provide a valid path to save the model.')
                
# Read data file
def read_data_file(x):
    df = pd.read_csv(x)
    df = df.drop('Unnamed: 0',1)
    df = df.drop('seconds', 1)
    return df

def normalize_data(df_list):
    ### **** indicates List of column names
    index_names = [****]
    min_max_values = np.array([0, 2040]).reshape((1,-1)).repeat(len(index_names),axis=0)
    min_max_values[6] = [0,3]
    mm_df = pd.DataFrame(min_max_values, columns=['min','max'], index=index_names)
    
    mod_list = [(item-mm_df['min'])/(mm_df['max']-mm_df['min']) for item in df_list]
    
    return mod_list


def unnormalize_data(tf_list):
    ### **** indicates List of column names
    index_names = [****]
    min_max_values = np.array([0, 2040]).reshape((1,-1)).repeat(len(index_names),axis=0)
    min_max_values[6] = [0,3]
    mm_df = pd.DataFrame(min_max_values, columns=['min','max'], index=index_names)
    
    df_list = []
    for item in tf_list:
        tmp_df = pd.DataFrame(item.numpy(), columns=index_names)
        tmp_df = tmp_df*(mm_df['max']-mm_df['min']) + mm_df['min']
        df_list.append(tmp_df)
    
    return df_list

#### Include the data in the CSV Format
files_list = ['***.csv', '***.csv']

for item in files_list:
    # **** indicates path of the filelist
    command = Popen(["hdfs", "dfs", "-get", '****'+item, item], stdin=PIPE, bufsize=-1)
    command.communicate()
    time.sleep(1)

#read in and process data
df_list = [read_data_file(item) for item in files_list]
norm_list = normalize_data(df_list)
data_list = [item.values for item in norm_list]

# Set model params
gen_dim = [32, 32, 32]
disc_dim = [32, 32, 32]
emb_dim = [32, 32, 32]
rec_dim = [32, 32, 32]
sup_dim = [32, 32]

# Set seed
reset_random_seeds()

# Create TimeGAN
tgan = TimeGAN(gen_dim, disc_dim, emb_dim, rec_dim, sup_dim, lr=0.0005)

# Train model
loss_dict = tgan.train(data_list, epochs=30000, pre_epochs=200, plot_loss=True)
