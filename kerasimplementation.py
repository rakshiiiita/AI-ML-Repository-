import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical

#step_2 Load the data
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

train_images=train_images.reshape((60000,28,28,1)).astype('float32')/255
tests_images=test_images.reshape((10000,28,28,1)).astype('float32')/255

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#step3 build
model=Sequential()
model.add(Dense(units=512,activation='relu',input_shape=(28*28,)))
model.add(Dense(units=10,activation='softmax'))


#step4 compile
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#history list
history=model.fit(train_images.reshape((60000,28*28)),train_labels,epochs=10,batch_size=128,validation_data=(test_images.reshape((10000,28*28)),test_labels))
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper left')

#show plots
plt.tight_layout()
plt.show()