#cd /mnt/c/temp/TensorFlow/python_pg/4_0_Attention_future
#python Attention_learning.py
#近未来推定用

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np 
import pandas as pd

import typing
from typing import Any, Tuple

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf

import textwrap

#parameter////////////////////////////////////////////////////////////////////////////////////////

TRAIN_CSV_PATH = "./seed8.csv"
TEST_CSV_PATH = "./seed9.csv"
NET_SAVE_PATH = "./pretrained_net.tf"
TIME_STEP = 0.002
ITERATION_NUM = 10#予測ステップ数

UNITS = 128#学習モデルの隠れ層数
FEATURE_SIZE = 1#入出力特徴量
BATCH_SIZE = 1012#バッチサイズ
EPOCHS = 2#最大エポック数
LEARNING_LATE = 0.0001#学習率
SEQUENCE_LEN = 512#シーケンス長
BATCH_STEP = 256#バッチのステップごとの移動
FIT_NUM = range(0,2)#バッチステップ反復回数インデント超えないように注意


PREDICT_SIZE = 2000#予測サイズ
#COLOR_LEVELS = np.array(range(0,40))*0.0003#カラーマップの分割数
COLOR_LEVELS = 40#カラーマップの分割数一貫させると収まりきらなくなるので


#def////////////////////////////////////////////////////////////////////////////////////////

#Tensorの配列形状のチェッククラス
class ShapeChecker():
  def __init__(self):
    # Keep a cache of every axis-name seen
    self.shapes = {}
#
  def __call__(self, tensor, names, broadcast=False):
    if not tf.executing_eagerly():
      return
#
    parsed = einops.parse_shape(tensor, names)
#
    for name, new_dim in parsed.items():
      old_dim = self.shapes.get(name, None)
#
      if (broadcast and new_dim == 1):
        continue
#
      if old_dim is None:
        # If the axis name is new, add its length to the cache.
        self.shapes[name] = new_dim
        continue
#
      if new_dim != old_dim:
        raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                         f"    found: {new_dim}\n"
                         f"    expected: {old_dim}\n")

#エンコーダレイヤ、GRUを内包し、stateと各出力を計算
class Encoder(tf.keras.layers.Layer):
  def __init__(self,normalizer ,feature_size, units):
    super(Encoder, self).__init__()
    self.normalizer = normalizer
    self.feature_size = feature_size
    self.units = units
#
    # The embedding layer converts tokens to vectors
    #マスクはおそらくいらない
    #ユニット数に合わせて入力を分解
    self.Dense = tf.keras.layers.Dense(self.units)
#
    # The RNN layer processes those vectors sequentially.
    #双方向は使えないと思って抜いたけど意外と使えるかも
    self.rnn = tf.keras.layers.GRU(units,
                                  # Return the sequence and state
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
#
  def call(self, x):
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch s feature_size')
#
    # 2. The embedding layer looks up the embedding vector for each token.
    x = self.Dense(x)
    shape_checker(x, 'batch s units')
#
    # 3. The GRU processes the sequence of embeddings.
    x, state = self.rnn(x)
    shape_checker(x, 'batch s units')
#
    # 4. Returns the new sequence of embeddings.
    return x, state
#
  #正規化関数selfでcallを読んでるので、この関数でエンコーダになる様子
  def normalize(self, data):
    context = self.normalizer(data)
    context, state = self(context)
    return context, state

#アテンションレイヤ、デコーダの出力とエンコーダの出力を用いてattentionベクトルを作成
class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__()
    #num_headsってなんだ？入力の数っぽい。
    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
    #正規化レイヤが内包されてる。この形式でも使えるのか？正規化の解除に困るか
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
#
  def call(self, x, context):
    shape_checker = ShapeChecker()
#
    shape_checker(x, 'batch t units')
    shape_checker(context, 'batch s units')
#
    #ここでおそらく各場合でのスコアと加重平均を取ったベクトルが出てきている
    attn_output, attn_scores = self.mha(
        query=x,
        value=context,
        return_attention_scores=True)
#
    shape_checker(x, 'batch t units')
    shape_checker(attn_scores, 'batch heads t s')
#
    # Cache the attention scores for plotting later.
    #各デコーダおよびエンコーダステップでのスコアの平均を取ってきている。
    #これをプロットしてやれば数値同士の関連度がわかる
    attn_scores = tf.reduce_mean(attn_scores, axis=1)
    shape_checker(attn_scores, 'batch t s')
    self.last_attention_weights = attn_scores
#
    #ここで、デコーダ出力とアテンションを加算している
    x = self.add([x, attn_output])
    #正規化はやる必要あるかはわからん
    x = self.layernorm(x)
#
    return x

#デコーダレイヤ、1ステップ前の予測をもとにイメージベクトルを出力,アテンションレイヤを内包
class Decoder(tf.keras.layers.Layer):
  ##@classmethodはデコレータの一種。本来は関数を書き換えてやる機能だが、クラスメソッドはクラスのインスタンスをする前に実行できる。
  ## ex"Decoder.add_method(関数名)"みたいに。これチュートリアルの途中で関数を追加できるようにするためのメソッドだ。
  #@classmethod
  #def add_method(cls, fun):
  #  setattr(cls, fun.__name__, fun)
  #  return fun
#
  def __init__(self, feature_size, units):
    super(Decoder, self).__init__()
    self.feature_size = feature_size
    self.start_token = tf.constant(np.zeros([feature_size]),dtype = "float32")
    self.units = units
#
#
    # 1. The embedding layer converts token IDs to vectors
    self.Dense = tf.keras.layers.Dense(self.units)
#
    # 2. The RNN keeps track of what's been generated so far.
    self.rnn = tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
#
    # 3. The RNN output will be the query for the attention layer.
    self.attention = CrossAttention(units)
#
    # 4. This fully connected layer produces the logits for each
    # output token.
    self.output_layer = tf.keras.layers.Dense(self.feature_size)
#
  def call(self,
          context, x,
          state=None,
          return_state=False):  
    shape_checker = ShapeChecker()
    shape_checker(x, 'one t feature_size')
    shape_checker(context, 'one s units')
#
    # 1. Lookup the embeddings
    x = self.Dense(x)
    shape_checker(x, 'one t units')
#
    # 2. Process the target sequence.
    x, state = self.rnn(x, initial_state=state)
    shape_checker(x, 'one t units')
#
    # 3. Use the RNN output as the query for the attention over the context.
    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    shape_checker(x, 'one t units')
    shape_checker(self.last_attention_weights, 'one t s')
#
    # Step 4. Generate logit predictions for the next token.
    logits = self.output_layer(x)
    shape_checker(logits, 'one t feature_size')
#
    if return_state:
      return logits, state
    else:
      return logits
#
#初期入力用の関数:予測的にはバッチは１
  def get_initial(self):
    #バッチサイズ調べて、そのサイズ分の0行列を作ってる
    start_tokens = tf.fill([1, 1, 1], self.start_token)
    return start_tokens
#
#予測関数、エンコーダ出力、前回の出力、前回の状態量。translate関数がめんどくさくなりそう
  def get_next_token(self, context, next_token, state):
    logits, state = self(
      context, next_token,
      state = state,
      return_state=True) 
#
    next_token = logits
#
    return next_token, state

#統括レイヤ、エンコーダ、デコーダを内包
class Translator(tf.keras.Model):
  #@classmethod
  #def add_method(cls, fun):
  #  setattr(cls, fun.__name__, fun)
  #  return fun
#
  def __init__(self, units,
               normalizer,
               feature_size):
    super().__init__()
    # Build the encoder and decoder
    encoder = Encoder(normalizer ,feature_size, units)
    decoder = Decoder(feature_size, units)
#
    self.encoder = encoder
    self.decoder = decoder
    self.units = units
#
  def call(self, in_data):
    context, x = in_data
    context, state = self.encoder(context)
    logits = self.decoder(context, x, state)
#
    return logits
#
#予測関数、predictだと思えばいい
#単一シーケンスデータを用いて予測、前回の状態は入れてやる
  def predict_one_step(self,in_data,before_token):
    # Process the input texts
    #
    if tf.shape(in_data)[0]==1:
      context, state = self.encoder.normalize(in_data)
  #
      # Generate the next token
      token, state = self.decoder.get_next_token(context, before_token, state)
#
      attention_weight = self.decoder.last_attention_weights
#
      return token, attention_weight
    else:
      print("Batch Size need to be 1")
#
#複数データも行けるpredict
  def predict(self,in_data):
    # Process the input texts
    context, states = self.encoder.normalize(in_data)
    batch_size = tf.shape(in_data)[0]
#
    # Setup the loop inputs
    tokens = []
    attention_weights = []
    initial_token = self.decoder.get_initial()
    next_token = initial_token
#
    for i in range(batch_size):
      # Generate the next token
      #next_token, state = self.decoder.get_next_token(tf.constant(np.array(context[i,:,:]).reshape((1,-1,self.units))), next_token, tf.constant(np.array(states[i,:]).reshape((1,self.units))))
      next_token, state = self.decoder.get_next_token(tf.constant(np.array(context[i,:,:]).reshape((1,-1,self.units))), initial_token, tf.constant(np.array(states[i,:]).reshape((1,self.units))))
#
      # Collect the generated tokens
      tokens.append(next_token)
      attention_weights.append(self.decoder.last_attention_weights)
#
    # Stack the lists of tokens and attention weights.
    tokens = tf.squeeze(tokens)   # t*[(batch 1)] -> (batch, t)
    self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)
#
    result = tokens
    return result

#クラスに格納するとメモリがオーバーフローしそうなのでこっちに分割
def predict_longseq(model, in_data):
  data_size = tf.shape(in_data)[0]
#
  tokens = []
  attention_weights = []
  token = model.decoder.get_initial()
#
  print("Predict")
#
  for i in range(data_size):
    token,attention_weight = model.predict_one_step(tf.constant(np.array(in_data[i,:,:]).reshape((1,-1,model.encoder.feature_size))), token)
    tokens.append(token)
    attention_weights.append(attention_weight)
    if i%50==0:
      print("/",end = "")
  #
  tokens = tf.squeeze(tokens)
  attention_weights = tf.squeeze(attention_weights)
  print("\nFinish")
#
  return tokens, attention_weights
  
#配列データをnumpyのarrayに変換
def make_data(in_data,out_data,sequence_len,iteration_num):
  out_sequences = []
  in_sequences = []
  for i in range(len(in_data) - sequence_len - iteration_num): 
    out_sequences.append([out_data[i + sequence_len + iteration_num]]) 
    in_sequences.append([in_data[i:i + sequence_len]])
  out_sequences = np.array(out_sequences)
  in_sequences = np.array(in_sequences).reshape((-1, sequence_len, 1))
  return [out_sequences,in_sequences]

#main////////////////////////////////////////////////////////////////////////////////////////

#学習データの作成------------------------------------------------------------------------------
train_csv_data = pd.read_csv(TRAIN_CSV_PATH)
test_csv_data = pd.read_csv(TEST_CSV_PATH)

train_x0_data = train_csv_data['x0'].values
train_x1_data = train_csv_data['x1'].values
test_x0_data = test_csv_data['x0'].values
test_x1_data = test_csv_data['x1'].values

#配列をarrayに変換
[train_out_seq,train_in_seq] = make_data(train_x1_data,train_x1_data,SEQUENCE_LEN,ITERATION_NUM)
[test_out_seq,test_in_seq] = make_data(test_x1_data,test_x1_data,SEQUENCE_LEN,ITERATION_NUM)

#Tensorに変換
train_out_seq = tf.constant(train_out_seq)
train_in_seq = tf.constant(train_in_seq)
test_out_seq = tf.constant(test_out_seq)
test_in_seq = tf.constant(test_in_seq)

#正規化レイヤーの作成ここからmeanとvarianceとってきてあげればアンパックができる
normalizer_in = tf.keras.layers.Normalization(axis = -1)
normalizer_out = tf.keras.layers.Normalization(axis = -1)

normalizer_in.adapt(train_in_seq)
normalizer_out.adapt(train_out_seq)

#レイヤーの作成------------------------------------------------------------------------------

model = Translator(UNITS, normalizer_in, FEATURE_SIZE)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_LATE),
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

for j in range(EPOCHS):
#
  print("EPOCH:",end="")
  print(j,end="")
  print("/",end="")
  print(EPOCHS)
#
  for i in FIT_NUM:
#
    print(i,end="")
    print("STEP")
    #学習用データの作成
    train_enc_origin = train_in_seq[i*BATCH_STEP:(i*BATCH_STEP)+BATCH_SIZE,:,:]
    train_enc_in = normalizer_in(train_enc_origin)
#
    #inisialize_x = np.zeros((1,FEATURE_SIZE))
    #train_dec_in = np.concatenate([inisialize_x,test_x1_seq[:BATCH_SIZE-1,:]],0)
    #train_dec_in = np.array(train_dec_in)
    #train_dec_in = tf.reshape(train_dec_in,[BATCH_SIZE,1,FEATURE_SIZE])
    train_dec_in = tf.zeros((BATCH_SIZE,1,FEATURE_SIZE))
#
    train_ans = normalizer_out(train_out_seq)
    train_ans = tf.reshape(train_ans[i*BATCH_STEP:(i*BATCH_STEP)+BATCH_SIZE,:],[BATCH_SIZE,1,FEATURE_SIZE])
#
    #学習
    history = model.fit(
        [train_enc_in,train_dec_in], 
        train_ans,
        epochs=1,
        #callbacks=[
            #tf.keras.callbacks.EarlyStopping(monitor='root_mean_squared_error',patience=3)]
            )
#
    #history = model.fit(a, b, epochs=EPOCHS)


#predict and plot ///////////////////////////////////////////////////////////////////////////////////////////////////////
test1,weight_attention1 = predict_longseq(model,train_in_seq[:PREDICT_SIZE,:,:])
test1 = (np.array(test1) * np.sqrt(normalizer_out.variance)) + normalizer_out.mean
test1 = np.reshape(test1,(PREDICT_SIZE))

test = test1

time = np.zeros(PREDICT_SIZE)
for i in range(0,PREDICT_SIZE):
  time[i] = (i+SEQUENCE_LEN+ITERATION_NUM)*TIME_STEP

step = np.array(range(0, SEQUENCE_LEN))*TIME_STEP


plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (4+4*(PREDICT_SIZE/SEQUENCE_LEN), 6)

fig = plt.figure()
plt.plot(time,test,label='predict')
plt.plot(time,train_in_seq[:PREDICT_SIZE,-1],label='answer')
plt.xlabel('time(s)')
plt.ylabel('x1(mm)')
plt.grid()
plt.legend()
fig.savefig("./train_predict.png")

plt.rcParams["figure.figsize"] = (4+4*(PREDICT_SIZE/SEQUENCE_LEN), 12)
train_fig_map, train_ax = plt.subplots(2,1,sharex = True ,tight_layout=True)
con = train_ax[1].contourf(time, step, np.array(weight_attention1).T[::-1,:],levels=COLOR_LEVELS)
train_ax[0].plot(time,train_in_seq[:PREDICT_SIZE,-1],label='input')
train_ax[0].plot(time,test,label='predict')
train_fig_map.colorbar(con, orientation='horizontal')
#train_ax[1].set_aspect('equal')
train_ax[1].set_title("train attention")
train_ax[0].set_title("predict")
train_ax[1].set_ylabel("sequence time(s)")
train_ax[1].set_xlabel("time(s)")
train_ax[0].set_ylabel("x1(mm)")
train_ax[0].grid()
train_ax[0].legend()
train_fig_map.savefig("./train_map.png")


test2,weight_attention2 = predict_longseq(model,test_in_seq[:PREDICT_SIZE,:,:])
test2 = (np.array(test2) * np.sqrt(normalizer_out.variance)) + normalizer_out.mean
test2 = np.reshape(test2,(PREDICT_SIZE))
test = test2

plt.rcParams["figure.figsize"] = (4+4*(PREDICT_SIZE/SEQUENCE_LEN), 6)
fig2 = plt.figure()
plt.plot(time,test,label='predict')
plt.plot(time,test_in_seq[:PREDICT_SIZE,-1],label='answer')
plt.xlabel('time(s)')
plt.ylabel('x1(mm)')
plt.grid()
plt.legend()
fig2.savefig("./test_predict.png")

plt.rcParams["figure.figsize"] = (4+4*(PREDICT_SIZE/SEQUENCE_LEN), 12)
test_fig_map, test_ax = plt.subplots(2,1,sharex = True, tight_layout=True)
con = test_ax[1].contourf(time, step, np.array(weight_attention2).T[::-1,:],levels=COLOR_LEVELS)
test_ax[0].plot(time,test_in_seq[:PREDICT_SIZE,-1],label='input')
test_ax[0].plot(time,test,label='predict')
test_fig_map.colorbar(con, orientation='horizontal')
#test_ax[1].set_aspect('equal')
test_ax[1].set_title("test attention")
test_ax[0].set_title("predict")
test_ax[1].set_ylabel("sequence time(s)")
test_ax[1].set_xlabel("time(s)")
test_ax[0].set_ylabel("x1(mm)")
test_ax[0].grid()
test_ax[0].legend()
test_fig_map.savefig("./test_map.png")

np.savetxt('./test_csv/time.csv',time,delimiter=',')
np.savetxt('./test_csv/step.csv',step,delimiter=',')
np.savetxt('./test_csv/predict.csv',test2,delimiter=',')
np.savetxt('./test_csv/answer.csv',np.array(test_out_seq[:PREDICT_SIZE]),delimiter=',')
np.savetxt('./test_csv/weight_attention.csv',np.array(weight_attention2),delimiter=',')

#保存ができるか試してる
try:
  model.save('./my_model.h5')

  #plt.plot(predict, label='predict')
  #plt.plot(tf.reshape(test_x1_seq[:BATCH_SIZE,:],[BATCH_SIZE]), label='answer')
  #plt.xlabel('step(-)')
  #plt.ylabel('x1(mm)')
  #plt.legend()
  #plt.show()


  # Restore the weights


  new_model = tf.keras.models.load_model('my_model')
except:
  print("Can't open file\n")
  print("Oh my god")