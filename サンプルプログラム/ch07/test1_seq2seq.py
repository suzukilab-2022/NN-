# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel


class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):#重みの初期化
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        #追加
        lstm_Wx2 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        #追加
        self.lstm2 = TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=False)

        self.params = self.embed.params + self.lstm.params #メンバ変数 params(重みパラメータ)
        self.grads = self.embed.grads + self.lstm.grads #メンバ変数 grads(勾配)
        #追加
        self.params2 = self.embed.params + self.lstm2.params #メンバ変数 params(重みパラメータ)
        self.grads2 = self.embed.grads + self.lstm2.grads #メンバ変数 grads(勾配)

        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        
        hs = self.lstm.forward(xs)
        #追加
        hs = self.lstm2.forward(hs)

        self.hs = hs
        return hs[:, -1, :]

    def backward(self, dh , dh2):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh
        
        #追加
        dhs2 = np.zeros_like(self.hs)
        dhs2[:, -1, :] = dh2

        #追加
        dout = self.lstm2.backward(dhs2)

        dout = self.lstm.backward(dhs)

        dout = self.embed.backward(dout)
        return dout


class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')  
        #追加
        lstm_Wx2 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        #追加
        self.lstm2 = TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True)

        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.lstm2, self.affine): #self.lstm2を追加
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.lstm.set_state(h)
        self.lstm2.set_state(h) #追加

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        #追加
        out = self.lstm2.forward(out)

        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
       
        dout = self.lstm2.backward(dout) #追加
        dout = self.lstm.backward(dout)
        
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        dh2 = self.lstm2.dh #追加 
        return dh , dh2

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)
        #追加
        self.lstm2.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            out = self.lstm2.forward(out) #追加
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled


class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.encoder.params2 + self.decoder.params #self.encoder.params2を追加 メンバ変数 params(重みパラメータ)
        self.grads = self.encoder.grads + self.encoder.grads2 + self.decoder.grads #self.encoder.grads2を追加 メンバ変数 grads(勾配)

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):

        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
