import tensorflow as tf
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify

#Start a Tensorflow session
sess = tf.Session()

# This is the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]), name = 'W')
b = tf.Variable(tf.zeros([10]), name = 'b')
y = tf.nn.softmax(tf.matmul(x, W) + b)

#Load the stored model
saver = tf.train.Saver()
saver.restore(sess, "model/trained.ckpt")

#Create the Flask app
app = Flask(__name__)

#A request handler for calling the Tensorflow model
@app.route('/', methods=['POST'])
def process_call():
    #prepare the model
    input = ((255 - np.array(request.get_json(), dtype=np.uint8)) / 255.0).reshape(1, 784)
    #execute the model and respond the result as JSON
    return jsonify(
        sess.run(y, feed_dict={x: input}).flatten().tolist()
    )

if __name__ == '__main__':
    app.run(debug=True)
    