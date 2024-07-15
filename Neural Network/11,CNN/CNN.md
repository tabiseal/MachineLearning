# 畳み込みニューラルネットワーク（CNN）


LeNet-5畳み込みニューラルネットワークを構築する

MNISTデータセットのグレースケール画像には1つのチャネルデータしかありません。したがって、畳み込みカーネルの3次元値は画像のチャネル数と一致する必要があります。

グレースケール画像には1つのチャネルしかないため、データを抽出した後に以下のように処理する必要があります。

```python
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

X_train=X_train.reshape(60000,28,28,1)/255.0
X_test=X_test.reshape(10000,28,28,1)/255.0
#60000枚の図,大きさ28*28
#チャンネル数:1;
#/255.0正規化操作

```

LeNet-5の入力は32×32のデータで、5×5×6の畳み込みカーネルを通じて28×28×6になります。

MNISTデータセットの場合、元のサイズは28×28×1で、5×5×6の畳み込みカーネルを通じて24×24×6になります。

**出力の数値0-9を10個のone-hotエンコードに変換します。**

```python
from keras.utils import to_categorical
Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)

```

**畳み込み層を実装する方法：**

```python
from keras.layers import Conv2D
model=Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5) ,strides=(1,1),input_shape=(28,28,1), padding='valid',activation='relu'))
#filters畳み込みフィルタ
#kernel_size畳み込みのサイズ
#stridesストライド(畳み込みフィルタ毎回移動する長さ)
#input_shape入力形状
#paddingパディングモード
#activation活性化関数
```

最初の24×24×6の畳み込み層を得た後、その結果をプーリング層に送ります。LeNet-5では、ウィンドウサイズが2×2の平均プーリングを使用します。

**2次元の平均プーリング層を使用する方法：**

```python
from keras.layers import AveragePooling2D
model.add(AveragePooling2D(pool_size=(2,2)))
#pool_sizeサイズ
```

池化層を通過すると、サイズが12×12の画像が6枚得られます。

各5×5の畳み込みカーネルは、6つの異なる特徴画像上でそれぞれ線形演算を行い、その結果を合計してサイズが8×8の新しい2次元特徴画像を生成します。このプロセスは16回繰り返されるため、16枚の8×8の特徴画像が得られます。

**さらに、もう一層のプーリング層を追加します。**

```python
model.add(AveragePooling2D(pool_size=(2,2)))
```

16枚の8×8の特徴画像をプーリング層を通過させると、16枚の2×2の特徴画像になります。

次に、16枚の2×2の特徴画像をフラット化して1つの配列にし、その配列を全結合層に接続します。

**方法：**

```python
from keras.layers import Flatten
model.add(Flatten())
```

**ニューラルネットワーク**

```python
model.add(Dense(units=120,activation='relu'))
model.add(Dense(units=84,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
```

活性化関数：'relu'関数

'softmax'関数（多クラス分類問題に適しています）

**訓練**

```python
model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.05),metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=500,batch_size=128)
```

コスト関数：「クロスエントロピーコスト関数」

アルゴリズム：「確率的勾配降下法（SGD）」

結果の選択

```python
import random
index = random.randint(0, len(X_test) - 1)
sample_image = X_test[index]
true_label = np.argmax(Y_test[index])

# モデルを使用して予測する
predicted_label = np.argmax(model.predict(sample_image.reshape(1, 28, 28, 1)))

# 画像を描く
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
plt.axis('off')
plt.show()

```


![image](./predictResult2.png)

