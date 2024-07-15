import random
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()


X_train=X_train.reshape(60000,28,28,1)/255.0
X_test=X_test.reshape(10000,28,28,1)/255.0
#60000枚の図/大きさ28*28/チャンネル数:1;/255.0正規化操作

Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)

model=Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5) ,strides=(1,1),input_shape=(28,28,1), padding='valid',activation='relu'))
#filters畳み込みフィルタ
#kernel_size畳み込みのサイズ
#stridesストライド(畳み込みフィルタ毎回移動する長さ)
#input_shape入力形状
#paddingパディングモード
#activation活性化関数
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(5,5) ,strides=(1,1), padding='valid',activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Flatten())


model.add(Dense(units=120,activation='relu'))
model.add(Dense(units=84,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.05),metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=500,batch_size=2048)


loss,accuracy=model.evaluate(X_test,Y_test)
print("loss"+str(loss))
print("accuracy"+str(accuracy))
# loss0.04554622620344162
# accuracy0.9865000247955322

index = random.randint(0, len(X_test) - 1)
sample_image = X_test[index]
true_label = np.argmax(Y_test[index])

# 使用模型进行预测
predicted_label = np.argmax(model.predict(sample_image.reshape(1, 28, 28, 1)))

# 绘制图像
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
plt.axis('off')
plt.show()

# 打印预测结果和是否正确
if true_label == predicted_label:
    print("预测结果正确")
else:
    print("预测结果不正确")
