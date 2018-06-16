import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
from lstm import LSTM
from controller import Controller


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=10,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=200,
                        help='save_old frequency')
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding dimension for the spatial coordinates')
    args = parser.parse_args()

    controller = Controller(dt=0.05, u_max=1., radius=1.22, to_point=True, no_random=False, render=False,
                              num_batches=10, batch_size=50, seq_length=10, xy_shift=2, target=[0., -0.5], reached=0.04)
    controller.P_init = [1., -0.7]
    # Save the arguments int the config file
    with open( os.path.join('save_lstm', 'config.pkl'), 'wb' ) as f:
        pickle.dump(args, f)
    # Create a Vanilla LSTM model with the arguments
    model = LSTM(args)
    # Initialize a TensorFlow session
    sess = tf.Session()
    # Initialize all the variables in the graph
    sess.run(tf.initialize_all_variables())
    # Add all the variables to the list of variables to be saved
    saver = tf.train.Saver(tf.all_variables())

    # For each epoch
    for e in range(args.num_epochs):
        # Assign the learning rate (decayed acc. to the epoch number)
        sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
        # Get the initial cell state of the LSTM
        state = sess.run(model.initial_state)
        # For each batch in this epoch
        batches = 11  # data_loader.num_batches
        for b in range(batches):
            # Tic
            start = time.time()
            # Get the source and target data of the current batch
            # x has the source data, y has the target data
            x, y = controller.next_batch()
            # Feed the source, target data and the initial LSTM state to the model
            feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
            # Fetch the loss of the model on this batch, the final LSTM state from the session
            train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
            # Toc
            end = time.time()
            # Print epoch, batch, loss and time taken
            print(
                "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                    e * batches + b,
                    args.num_epochs * batches,
                    e,
                    train_loss, end - start))

            # Save the model if the current epoch and batch number match the frequency
            if (e * batches + b) % args.save_every == 0 and ((e * batches + b) > 0):
                checkpoint_path = os.path.join('save_lstm', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e * batches + b)
                print("model saved to {}".format(checkpoint_path))
    sess.close()

if __name__ == '__main__':
    main()
