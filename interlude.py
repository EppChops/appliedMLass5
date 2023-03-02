from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input

vggmodel = applications.VGG16(weights='imagenet', include_top=True)


img = load_img("a5_images/train/car/0000.jpg", target_size=(224,224))

arr = img_to_array(img)

preprocessed = preprocess_input(arr)

preprocessed = preprocessed.reshape(1, 224,224, 3)

pred = vggmodel.predict(preprocessed)

print(decode_predictions(pred))