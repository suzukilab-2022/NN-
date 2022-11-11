# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence #sequence.py　id_to_char = {} char_to_id = {}の初期化
from common.optimizer import Adam #optimizer.pyの101行目 adamという最適化アルゴリズムの一種
from common.trainer import Trainer #trainer.py　
from common.util import eval_seq2seq #util.pyの227行目 正解率を計測
from seq2seq import Seq2seq #同ファイル seq2seqの処理(encoder,decoderなど)
from peeky_seq2seq import PeekySeq2seq #同ファイル peeky(覗き見)の処理


# データセットの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')#adition.txtが学習データ
char_to_id, id_to_char = sequence.get_vocab() #id_to_char = {} char_to_id = {}をしている


# Reverse input?(reverse(データ入力の反転)を実行するか)=====================================
is_reverse = False  # True
if is_reverse: # trueであったら実行されreverseを行う
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
# ================================================================

# ハイパーパラメータ(推論や予測の枠組みの中で決定されないパラメータのこと)の設定
vocab_size = len(char_to_id) #語彙数 len() 引数に指定したオブジェクトの長さや要素数を取得
wordvec_size = 16 #文字ベクトルの次元数
hidden_size = 128 #LSTMレイヤの隠れ状態ベクトルの次元数
batch_size = 128 #p49　ミニバッチのサイズ
max_epoch = 25 #p49 学習を行うエポック数
max_grad = 5.0 #p49　勾配の最大ノルム


#モデル/オプティマイザ/トレーナーの生成
#tinderクラスの実行
# Normal or Peeky? ==============================================
model = Seq2seq(vocab_size, wordvec_size, hidden_size) #通常
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size) #peeky(覗き見)の処理も実行する
# ================================================================
optimizer = Adam() 
trainer = Trainer(model, optimizer) 

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad) 

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse)

    acc = float(correct_num) / len(x_test) #acc = accurary(正答率)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

# グラフの描画
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.show()

