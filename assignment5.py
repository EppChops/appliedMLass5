from tensorflow.keras import applications, datasets, layers, models, losses
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input
from matplotlib import pyplot as plt
import numpy as np

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
# Xbatch, Ybatch = train_generator.next()

# print(Xbatch.shape)
# print(Ybatch.shape)

# print("annotation image #4:", Ybatch[4])

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
#print(history.history['val_accuracy'][len(history.history['val_accuracy'])-1])

augmented_gen = ImageDataGenerator(rescale=1.0/255, 
                                   rotation_range=10, 
                                   vertical_flip=True,
                                   brightness_range=(0.8, 1.2),
                                   shear_range=5
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
history = cnn.fit(augmented_train_generator, epochs=10, validation_data=validation_generator)

#cnn.save_weights('augmented_weights')
print(history.history['val_accuracy'][len(history.history['val_accuracy'])-1])

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(history.history['loss'])
# ax1.plot(history.history['val_loss'])
# ax1.set_title('model loss')
# ax1.set_ylabel('loss')
# ax1.set_xlabel('epoch')
# ax1.legend(['train', 'val'], loc='upper left')

# ax2.plot(history.history['accuracy'])
# ax2.plot(history.history['val_accuracy'])
# ax2.set_title('model accuracy')
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('accuracy')
# ax2.legend(['train', 'val'], loc="upper left")
# plt.show()

# **Accuracy of final epoch:** 0.7725694179534912

"""
feature_extractor = applications.VGG16(include_top=False, weights='imagenet',
                                        input_shape=(img_size, img_size, 3))

def create_vgg16_features(input_dir, out_file):

    vgg_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = vgg_data_gen.flow_from_directory(
            imgdir + input_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            classes=['other', 'car'],
            seed=12345,
            shuffle=False)

    cnn_features = feature_extractor.predict(train_generator)

    with open(out_file, 'wb') as f:
        np.save(f, cnn_features)

create_vgg16_features('/train', 'training_set')
create_vgg16_features('/validation', 'validation_set')

def get_labels(n):
    return np.array([0]*(n//2) + [1]*(n//2))

def make_vgg_classifier():
  model = models.Sequential()
  model.add(layers.Flatten())
  model.add(layers.Dense(1000, activation='relu'))
  model.add(layers.Dense(num_classes, activation='sigmoid'))
  
  model.compile(optimizer='adam', 
                loss=losses.BinaryCrossentropy(from_logits=False, label_smoothing=0), 
                metrics=['accuracy'])
  
#   print(model.summary())
  return model






def train_on_cnn_features(train_set, val_set):
        with open(train_set, 'rb') as f:
                train_data = np.load(f)
        with open(val_set, 'rb') as f:
                val_data = np.load(f)

        print(train_data.shape)
        print(val_data.shape)
        cnn = make_vgg_classifier()

        history = cnn.fit(train_data, get_labels(len(train_data)), epochs = 10, 
                validation_data =(val_data, get_labels(len(val_data))))
        
        return history

       
hist = train_on_cnn_features('training_set', 'validation_set')

print(hist.history['val_accuracy'][len(hist.history['val_accuracy'])-1])


## ---- Part 4 ---- ##


def kernel_image(weights, i, positive):
    
    # extract the convolutional kernel at position i
    k = weights[:,:,:,i].copy()
    if not positive:
        k = -k
    
    # clip the values: if we're looking for positive
    # values, just keep the positive part; vice versa
    # for the negative values.
    k *= k > 0

    # rescale the colors, to make the images less dark
    m = k.max()
    if m > 1e-3:
        k /= m 

    return k

 
first_layer_weights = feature_extractor.get_weights()[0]
print(first_layer_weights.shape)

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(kernel_image(first_layer_weights, 0, True))
axarr[0,1].imshow(kernel_image(first_layer_weights, 0, False))
plt.show()
"""