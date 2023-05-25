import sys
sys.path.append('PokemonProject')
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.utils import plot_model
import matplotlib.pyplot as plt
import argparse


def run():
        # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Training Pokemon Data')

    # Add arguments to the parser
    parser.add_argument('--train_dir', type=str, help='Enter the directory of the data')
    parser.add_argument('--val_dir', type=str, help='Enter the validation directory')
    
    # Parse the arguments
    args = parser.parse_args()
    neuralnetwork_cnn=train(args.train_dir)
    evaluate(neuralnetwork_cnn,args.train_dir,args.val_dir)
    
    
    



def train(train_generator):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    filepath = "model.h5"
    ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    rlp = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1)
    image_size = (64, 64, 3)
    num_classes = len(train_generator.class_indices)
    neuralnetwork_cnn = cnn(image_size, num_classes)
    neuralnetwork_cnn.summary()
    plot_model(neuralnetwork_cnn, show_shapes=True) 
    return neuralnetwork_cnn

   

def evaluate(neuralnetwork_cnn,train_generator,val_generator):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    filepath = "model.h5"
    ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    rlp = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1)
    test_loss, test_acc = neuralnetwork_cnn.evaluate(train_generator)
    print('Test accuracy:', test_acc)  
    history = neuralnetwork_cnn.fit_generator(
    generator=train_generator, validation_data=val_generator,
    callbacks=[es, ckpt, rlp], epochs = 50, 
    )
    h = history.history
    # Visualizing loss
    plt.plot(h['loss'],'r',label='Loss')
    plt.plot(h['val_loss'],'b',label='Val Loss')
    plt.legend()
    plt.show()
    
def cnn(image_size, num_classes):
    classifier = Sequential()
    classifier.add(Conv2D(64, (5, 5), input_shape=image_size, activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(num_classes, activation = 'softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return classifier





if __name__ == '__main__':
    run()
