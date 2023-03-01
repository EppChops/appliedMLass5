from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models, losses
from matplotlib import pyplot as plt

data_gen = ImageDataGenerator(rescale=1.0/255)

imgdir = 'a5_images' # or wherever you put them...
img_size = 64
batch_size = 32

# Read images from directory
train_generator = data_gen.flow_from_directory(
        imgdir + '/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        seed=12345,
        shuffle=True)

#Generate first batch of training data
Xbatch, Ybatch = train_generator.next()

print(Xbatch.shape)
print(Ybatch.shape)

print("annotation image #4:", Ybatch[4])

#
#plt.imshow(Xbatch[4])
#plt.show()

input_shape = (img_size, img_size, 3)
num_classes = 1

def make_convnet():
  model = models.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
  activation='relu',
  input_shape=input_shape))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(layers.Conv2D(64, (5, 5), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(1000, activation='relu'))
  model.add(layers.Dense(num_classes, activation='sigmoid'))
  
  model.compile(optimizer='adam', 
                loss=losses.BinaryCrossentropy(from_logits=False, label_smoothing=0), 
                metrics=['accuracy'])
  
  print(model.summary())
  return model


# Read images from directory
validation_generator = data_gen.flow_from_directory(
        imgdir + '/validation',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        seed=12345,
        shuffle=True)

Xtest, YTest = validation_generator.next()
#cnn = make_convnet()
#history = cnn.fit(train_generator, epochs=10, validation_data=validation_generator)

#cnn.save_weights('weights')


augmented_gen = ImageDataGenerator(rescale=1.0/255, 
                                   rotation_range=90, 
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   #brightness_range=(10, 255),
                                   #channel_shift_range=20.5
                                   )
# Read images from directory
augmented_train_generator = augmented_gen.flow_from_directory(
        imgdir + '/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        seed=12345,
        shuffle=True)

cnn = make_convnet()
history = cnn.fit(augmented_train_generator, epochs=20, validation_data=validation_generator)

cnn.save_weights('augmented_weights')

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.set_title('model loss')
ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'val'], loc='upper left')

ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.set_title('model accuracy')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.legend(['train', 'val'], loc="upper left")
plt.show()