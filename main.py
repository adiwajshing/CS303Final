import keras
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import glob

img_size = 150 # set universal size for images -- images will also be grayscale
batch_size = 64  # set universal batch size -- images processed per step

# load a saved gender detection model
def load_gender_model ():
	model = make_gender_model()
	model.load_weights("gender_model.h5")
	return model

# create the CNN to detect gender
def make_gender_model ():

	model = Sequential()
	model.add(Conv2D(64, kernel_size=3, input_shape=(img_size, img_size, 1), activation="relu", padding='same'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(MaxPooling2D())
	#model.add(BatchNormalization())

	model.add(Conv2D(32, kernel_size=3, activation="relu", padding='same'))
	model.add(MaxPooling2D())
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(BatchNormalization())

	model.add(Conv2D(32, kernel_size=3, activation="relu", padding='same'))
	model.add(MaxPooling2D())

	model.add(Conv2D(32, kernel_size=3, activation="relu", padding='same'))
	model.add(MaxPooling2D())
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(200, activation='relu'))
	model.add(Dropout(0.1))

	model.add(Dense(1, activation='sigmoid'))#, activity_regularizer=regularizers.l1(0.01)))

	model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['binary_accuracy'])
	return model

# train a gender detection CNN
def train_gender_model ():

	model = load_gender_model()	# create the skeleton for the CNN

	# we create an image data generator that randomly modifies an image
	# so that we have better & more noisy data to train on
	# we will get better generalisation performance
	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.01,
        zoom_range=0.01,
        horizontal_flip=True)

	# generator will read pictures found in the specified directory
	# classes are detected based on the folder
	# wil create random batches of the data it finds
	train_generator = train_datagen.flow_from_directory(
        'CLEANED/gender',  # this is the target directory
        target_size=(img_size, img_size),  # all images will be resized to 150x150
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

	# this is a similar generator, for validation data
	validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'CLEANED/test_gender',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary')

	#callbacks=[EarlyStopping(patience=3, restore_best_weights=True), 
     #      ReduceLROnPlateau(patience=2), 
      #     ModelCheckpoint(filepath='gender_model_chk.h5', save_best_only=True)] 
	# finally, we train the model
	model.fit_generator(
        train_generator,
        steps_per_epoch=2048 // batch_size,
        epochs=3,
        validation_data=validation_generator,
        class_weight={0:1.5, 1: 1.0}
        #callbacks=callbacks,
       # validation_steps=500 // batch_size
       )

	# print out what each class means
	print("class indices = " + str(train_generator.class_indices))
	
	# save the model
	model.save_weights('gender_model.h5')  # always save your weights after training or during training
	return model

def get_gender_confusion_matrix (model):
	validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'CLEANED/test_gender',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary',
        shuffle=False)


	y_pred = model.predict_generator(validation_generator)
	y_pred = [ (1 if y[0] > 0.5 else 0) for y in y_pred]

	print(sum(abs(validation_generator.classes-y_pred)))

	print('Confusion Matrix')
	cm = confusion_matrix(validation_generator.classes, y_pred)
	print(cm)

	plt.imshow(cm, cmap=plt.cm.Blues)
	plt.xlabel("Predicted labels")
	plt.ylabel("True labels")
	plt.xticks([], [])
	plt.yticks([], [])
	plt.title('Gender Confusion Matrix')
	plt.colorbar()
	plt.show()

def load_age_model ():
	model = make_age_model()
	model.load_weights("age_model.h5")
	return model

def make_age_model ():

	model = Sequential()

	model.add(Conv2D(64, kernel_size=5, input_shape=(img_size, img_size, 1), activation="relu", padding="same"))
	model.add(MaxPooling2D())

	model.add(Conv2D(64, kernel_size=3, activation="relu", padding="same"))
	model.add(MaxPooling2D())

	model.add(Conv2D(64, kernel_size=3, activation="relu", padding="same"))
	model.add(MaxPooling2D())

	model.add(Conv2D(64, kernel_size=3, activation="relu", padding="same"))
	model.add(MaxPooling2D())

	model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.1))

	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(400, activation='relu'))
	#model.add(Dropout(0.1))

	model.add(Dense(5, activation='sigmoid'))

	model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

	return model

def train_age_model ():

	model = load_age_model()	

	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.01,
        zoom_range=0.01,
        horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(
        'CLEANED/age',  # this is the target directory
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')  

	validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'CLEANED/test_age',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

	classes = validation_generator.classes
	class_weights = class_weight.compute_class_weight('balanced', np.unique(classes), classes)
	print (class_weights)

	#callbacks=[EarlyStopping(patience=3, restore_best_weights=True), 
    #       ReduceLROnPlateau(patience=2), 
     #      ModelCheckpoint(filepath='age_model_chk.h5', save_best_only=True)] 

	model.fit_generator(
        train_generator,
        steps_per_epoch=2048 // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=512 // batch_size,
        #callbacks=callbacks,
        class_weight=class_weights
        )

	print("class indices = " + str(train_generator.class_indices))
	
	model.save_weights('age_model.h5')  # always save your weights after training or during training
	return model

def get_age_confusion_matrix (model):
	validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'CLEANED/test_age',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False)


	y_pred = model.predict_generator(validation_generator)
	y_pred = np.argmax(y_pred, axis=1)

	print('Confusion Matrix')
	cm = confusion_matrix(validation_generator.classes, y_pred)

	classes = validation_generator.classes
	class_weights = class_weight.compute_class_weight('balanced', np.unique(classes), classes)

	cm = [ [ round( float(cm[i][j]) * class_weights[i], 2) for j in range(0, len(cm[i]))] for i in range(0, len(cm))  ]
	print(cm)

	plt.imshow(cm, cmap=plt.cm.Blues)
	plt.xlabel("Predicted labels")
	plt.ylabel("True labels")
	#plt.set_xlabel(validation_generator.class_indices)
	plt.xticks([], [])
	plt.yticks([], [])
	plt.title('Weighted Age Confusion Matrix')
	plt.colorbar()
	plt.show()


# load the image we want to test

testImageNames = glob.glob("CUSTOM_TEST_IMAGES/*.*")
images = []

for imageName in testImageNames:
	img = image.load_img(imageName, target_size=(img_size, img_size), color_mode='grayscale')
	x = image.img_to_array(img)
	
	x = np.expand_dims(x, axis=0)

	images.append( x )
images = np.vstack(images) 

# gender detection -----------

gmodel = load_gender_model()
#get_gender_confusion_matrix(gmodel)

classes = gmodel.predict_classes(images, batch_size=batch_size)
print ("genders: ")
for i in range(0, len(classes)):
	if classes[i][0] == 1:
		print("person in image '" + testImageNames[i] + "' is male")
	else:
		print("person in image '" + testImageNames[i] + "' is female")

# age detection -----------


amodel = train_age_model()
get_age_confusion_matrix(amodel)

classMap = {0: 'Baby', 1: 'Child', 2: 'Elderly', 3: 'Middle Aged', 4: 'Young'}
classes = amodel.predict_classes(images, batch_size=batch_size)

print("ages: ")
for i in range(0, len(classes)):
	print("person in image '" + testImageNames[i] + "' is in the category: " + classMap[classes[i]])