import click
import wget
import shutil
import os
import glob
import numpy as np
from PIL import Image, ImageOps


class Data:
    """Class to load data for training and testing"""
    def __init__(self, dir, model, mode):
        self.dir = dir
        self.model = model # model1, model2 or model3
        self.mode = mode # train, test, user
        self.size = 28
        self.X, self.Y = self.create_datasets()

    def load_img(self, path, size):
        """Function to load imagesself
        Args:
            path: image path
            size: desired size of the image
        Returns:
            image as numpy array"""

        img = Image.open(path)
        old_size = img.size

        ratio = float(size/max(old_size))
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)
        delta_w = size - new_size[0]
        delta_h = size - new_size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img = ImageOps.expand(img, padding)
        return np.array(img)

    def create_datasets(self):
        """Function which creates arrays of images which can be fed into the models"""
        X = []
        Y = []

        if self.mode != 'user:':
            for j in range(43):
                path = os.path.join('images', self.dir, str(j).rjust(2, '0'), '*.ppm')
                files = glob.glob(path)
                for file in files:
                    X.append(self.load_img(file, self.size))
                    Y.append(j)

        else:
            path = os.path.join('images', self.dir, '*.ppm')
            files = glob.glob(path)
            for file in files:
                X.append(self.load_img(file, self.size))

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        if (self.model == 'model1') | ((self.model == 'model2')):
            X = X.flatten().reshape(X.shape[0], self.size*self.size*3)

        if self.model != 'model1':
            import pickle
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.externals import joblib

            if self.mode == 'train':
                enc = OneHotEncoder()
                Y = enc.fit_transform(Y.reshape(-1, 1)).toarray()
                joblib.dump(enc, 'models/' + self.model + '/saved/encoder.pkl')
            else:
                enc = joblib.load('models/' + self.model + '/saved/encoder.pkl')
                Y = enc.transform(Y.reshape(-1, 1)).toarray()

        return X, Y


@click.group()
def cli():
    pass


@cli.command()
def download():
    """Download all the data from the Traffic Signs Dataset
    and store it inside the 'images' folder"""
    from zipfile import ZipFile

    url = "http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip"

    if os.path.exists('images/train') & os.path.exists('images/test'):
        click.echo("Downloading German Traffic Signs Dataset")
    else:
        click.echo("Required folder structure does not exist")
        return

    if not os.path.isfile('./FullIJCNN2013.zip'):
        wget.download(url, 'FullIJCNN2013.zip')
    print('Download complete. Extracting files.')

    zip = ZipFile('FullIJCNN2013.zip')
    zip.extractall()
    print('Zip extracted')

    for file in os.listdir('FullIJCNN2013/'):
        if os.path.isdir('FullIJCNN2013/'+file):
            shutil.move('FullIJCNN2013' + '/' + file, 'images/train/' + file)
    print('Moved images to train')

    shutil.rmtree('FullIJCNN2013')
    #os.remove('./FullIJCNN2013.zip')

    print('Splitting data: train (80%) and test (20%)')
    source = "images/train/"
    dest = "images/test/"
    folders = os.listdir(source)

    for folder in folders:
        if not os.path.exists(dest + folder):
            os.makedirs(dest + folder)
        files = os.listdir(source+folder)
        for file in files:
            if np.random.rand(1) < 0.2:
                shutil.move(source + folder + '/' + file, dest + folder + '/'+ file)
    print('Done')

@cli.command()
@click.option('--model', '-m', default='model1', prompt="choose model",
              help='Choose between model1, model2 and model3')
@click.option('--dir', '-d', default='train', prompt='image directory',
              help="Directory of images")
def train(model, dir):
    """Function to train a model for German Traffic Sign Image Classification
    model = model1 Trains a Logistic regression model with sklearn
    model = model2 Trains a Logistic regression model with Tensorflow
    model = model3 Trains a LeNet 5 Convolutional network model with Tensorflow
    """
    print('Starting...')

    data = Data(dir, model, 'train')

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(data.X, data.Y, test_size=0.2)
    print('Loaded Data')
    print(data.X.shape)

    if model == 'model1':
        from sklearn.linear_model import LogisticRegression

        print('Training logistic regression with Scikit-learn')

        log_model = LogisticRegression(solver='lbfgs')
        log_model.fit(train_x, train_y)
        score = log_model.score(test_x, test_y)
        print('Score:', score)

        import pickle
        pickle.dump(log_model, open('models/model1/saved/model1.sav', 'wb'))

    elif model == 'model2':
        import tensorflow as tf

        print('Training logistic regression with TensorFlow')

        epochs = 700
        lr = 0.003
        num_input = data.X.shape[1]
        num_classes = data.Y.shape[1]

        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=[None, num_input], name='x_input')
        y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_input')

        with tf.name_scope('Weights'):
            W = tf.Variable(tf.truncated_normal(shape=[num_input, num_classes]), name='W')
            b = tf.Variable(tf.truncated_normal(shape=[num_classes]), name='b')

        with tf.name_scope('Output'):
            y = tf.add(tf.matmul(x, W), b)

        with tf.name_scope("Loss"):
            # calculating cost
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
        with tf.name_scope("Optimizer"):
            # optimizer
            # we use gradient descent for our optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

        with tf.name_scope('Accuracy'):
            prediction = tf.argmax(y, 1, name='predict')
            correct = tf.equal(prediction, tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                train_x, test_x, train_y, test_y

                _, loss = sess.run([optimizer, cost], feed_dict = {x: train_x, y_: train_y})

                acc = sess.run(accuracy, feed_dict = {x: train_x, y_: train_y})
                test_acc = sess.run(accuracy, feed_dict = {x: test_x, y_: test_y})

                if epoch % 100 == 0:
                    print('Epoch', epoch)
                    print('Training accuracy:', acc)
                    print('Test accuracy:', test_acc)
            print('Training finished')

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('Accuracy:', accuracy.eval({x: test_x, y_: test_y}))

            saver.save(sess, 'models/model2/saved/model2')

    else:
        import tensorflow as tf
        print('Training LeNet with TensorFlow')
        epochs = 10000
        lr = 0.0003
        num_classes = data.Y.shape[1]

        def conv_layer(input_data, input_channels, filter_size, num_filters):
            W = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, input_channels, num_filters], stddev=0.05))
            b = tf.Variable(tf.constant(0.05, shape=[num_filters]))

            layer = tf.nn.conv2d(input=input_data, filter=W, strides=[1, 1, 1, 1], padding='SAME')
            layer += b

            return layer

        def fully_connected(input_data, num_inputs, num_outputs):
            W = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
            b = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

            return tf.matmul(input_data, W) + b

        #Reset tensorflow graph
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='x_input')
        y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_input')

        print('Input:', x.get_shape())
        c1 = conv_layer(input_data=x, input_channels=3, filter_size=5, num_filters=6)
        print('C1:', c1.get_shape())
        s2 = tf.nn.pool(input=c1, window_shape=[2, 2], pooling_type='AVG', padding='SAME', strides=[2,2])
        print('S2:', s2.get_shape())
        a1 = tf.nn.relu(s2)
        print('A1:', a1.get_shape())

        c3 = conv_layer(input_data=a1, input_channels=6, filter_size=5, num_filters=16)
        print('C3:', c3.get_shape())
        s4 = tf.nn.pool(input=c3, window_shape=[2, 2], pooling_type='AVG', padding='SAME', strides=[2,2])
        print('S4:', s4.get_shape())
        a2 = tf.nn.relu(s4)

        c5 = conv_layer(input_data=a2, input_channels=16, filter_size=5, num_filters=120)
        print('C5:', c5.get_shape())

        num_features = c5.get_shape()[1:4].num_elements()
        flat = tf.reshape(c5, [-1, num_features])
        f6 = fully_connected(input_data=flat, num_inputs=num_features, num_outputs=84)
        print('F6:', f6.get_shape())
        a3 = tf.nn.relu(f6)

        output = fully_connected(input_data=a3, num_inputs=84, num_outputs=num_classes)
        print('Output:', output.get_shape())

        with tf.name_scope('Output'):
            y = tf.nn.softmax(output, name='y_output')

        with tf.name_scope("Loss"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
        with tf.name_scope("declaring_gradient_descent"):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

        with tf.name_scope('Accuracy'):
            prediction = tf.argmax(y, 1, name='predict')
            correct = tf.equal(prediction, tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):

                _, loss = sess.run([optimizer, cost], feed_dict = {x: train_x, y_: train_y})

                acc = sess.run(accuracy, feed_dict = {x: train_x, y_: train_y})
                test_acc = sess.run(accuracy, feed_dict = {x: test_x, y_: test_y})

                if epoch % 2000 == 0:
                    print('Epoch', epoch)
                    print('Training accuracy:', acc)
                    print('Test accuracy:', test_acc)
            print('Training finished')

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('Accuracy:', accuracy.eval({x: test_x, y_: test_y}))

            saver.save(sess, 'models/model3/saved/model3')


@cli.command()
@click.option('--model', '-m', default='model1', prompt="choose model",
              help='Choose between model1, model2 and model3')
@click.option('--dir', '-d', default='model1', prompt='image directory',
              help="Directory of images")
def test(model, dir):
    """Function to test trained models on unseen data"""

    print('Starting...')

    data = Data(dir, model, 'test')
    print('Loaded Data')
    print(data.X.shape)
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(data.X, data.Y,
                                                        test_size=0.2)

    print('Testing...')
    if model == 'model1':
        import pickle
        loaded_model = pickle.load(open('models/model1/saved/model1.sav', 'rb'))
        result = loaded_model.score(data.X, data.Y)
        print('Test score', result)

    elif model == 'model2':
        import tensorflow as tf
        sess = tf.Session()
        saver = tf.train.import_meta_graph('models/model2/saved/model2.meta')
        saver.restore(sess, tf.train.latest_checkpoint('models/model2/saved'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x_input:0")
        y_ = graph.get_tensor_by_name("y_input:0")
        feed_dict = {x: test_x, y_: test_y}

        accuracy = graph.get_tensor_by_name("Accuracy/accuracy:0")

        print('Accuracy:', sess.run(accuracy, feed_dict))

    else:
        import tensorflow as tf
        sess = tf.Session()
        saver = tf.train.import_meta_graph('models/model3/saved/model3.meta')
        saver.restore(sess, tf.train.latest_checkpoint('models/model3/saved'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x_input:0")
        y_ = graph.get_tensor_by_name("y_input:0")
        feed_dict = {x: test_x, y_: test_y}

        accuracy = graph.get_tensor_by_name("Accuracy/accuracy:0")

        print('Accuracy:', sess.run(accuracy, feed_dict))


@cli.command()
@click.option('--model', '-m', default='model1', prompt="choose model",
              help='Choose between model1, model2 and model3')
@click.option('--dir', '-d', default='model1', prompt='image directory',
              help="Directory of images")
def infer(model, dir):
    """Function to make predictions with trained models on unseen images"""

    def load_img(path, size):
        """Function to load imagesself
        Args:
            path: image path
            size: desired size of the image
        Returns:
            image as numpy array"""

        img = Image.open(path)
        old_size = img.size

        ratio = float(size/max(old_size))
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)
        delta_w = size - new_size[0]
        delta_h = size - new_size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img = ImageOps.expand(img, padding)
        return np.array(img)

    if model == 'model1':
        import pickle
        loaded_model = pickle.load(open('models/model1/saved/model1.sav', 'rb'))
        files = glob.glob(os.path.join('images', dir, '*.ppm'))
        print(len(files))
        for file in files:
            img = Image.open(file)
            img.show()

            img = load_img(file, 28)
            img = img.flatten().reshape(1, 28*28*3)
            print('Predicted class:',loaded_model.predict(img))

    elif model == 'model2':
        import tensorflow as tf
        sess = tf.Session()
        saver = tf.train.import_meta_graph('models/model2/saved/model2.meta')
        saver.restore(sess, tf.train.latest_checkpoint('models/model2/saved'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x_input:0")

        prediction = graph.get_tensor_by_name("Accuracy/predict:0")

        files = glob.glob(os.path.join('images', dir, '*.ppm'))

        for file in files:
            img = Image.open(file)
            img.show()

            img = load_img(file, 28)

            img = img.flatten().reshape(1, 28*28*3)
            print('Predicted class:', sess.run(prediction, feed_dict = {x: img}))

    else:
        import tensorflow as tf
        sess = tf.Session()
        saver = tf.train.import_meta_graph('models/model3/saved/model3.meta')
        saver.restore(sess, tf.train.latest_checkpoint('models/model3/saved'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x_input:0")

        prediction = graph.get_tensor_by_name("Accuracy/predict:0")

        files = glob.glob(os.path.join('images', dir, '*.ppm'))

        for file in files:
            img = Image.open(file)
            img.show()

            img = load_img(file, 28)
            img = np.expand_dims(img, 0)
            print('Predicted class:', sess.run(prediction, feed_dict = {x: img}))


if __name__ == '__main__':
    cli()
