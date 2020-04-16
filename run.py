# importing libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import random

def main(element):
    learn = tf.contrib.learn 
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # importing dataset using MNIST 
    # this is how mnist is used mnist contain test and train dataset 
    mnist = learn.datasets.load_dataset('mnist') 
    data = mnist.train.images 
    labels = np.asarray(mnist.train.labels, dtype = np.int32) 
    test_data = mnist.test.images 
    test_labels = np.asarray(mnist.test.labels, dtype = np.int32) 
    
    max_examples = 10000
    data = data[:max_examples] 
    labels = labels[:max_examples] 
    
    # displaying dataset using Matplotlib 
    def display(i): 
        img = test_data[i] 
        plt.title('label : {}'.format(test_labels[i])) 
        plt.imshow(img.reshape((28, 28))) 
        plt.show()
        
    # img in tf is 28 by 28 px 
    # fitting linear classifier 
    feature_columns = learn.infer_real_valued_columns_from_input(data) 
    classifier = learn.LinearClassifier(n_classes = 10,  
                                        feature_columns = feature_columns) 
    classifier.fit(data, labels, batch_size = 100, steps = 1000) 
    
    # Evaluate accuracy 
    classifier.evaluate(test_data, test_labels) 
    print("\nAccuracy is: "+str(classifier.evaluate(test_data, test_labels)["accuracy"])) 
    
    prediction = classifier.predict(np.array([test_data[element]],  
                                            dtype=float),  
                                            as_iterable=False) 
    print("prediction : {}, label : {}".format(prediction,  
        test_labels[element]) ) 
    
    if prediction == test_labels[element]: 
        display(element) 
    else:
        print("Prediction was Inacurate, train the model for more iterations to improve the accuracy.")

main(random.randrange(0, 10000, 1))