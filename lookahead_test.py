"""
Test for lookahead optimizer.
"""

import numpy as np
import tensorflow as tf

import lookahead


class LookaheadOptimizerTest(tf.test.TestCase):
    # pylint: disable=too-many-locals
    """LookaheadOptimizer Test."""

    def setUp(self):
        self.k = 5
        self.alpha = 0.5

        self.batch_num = 100
        self.input_size = 1000

    def test_lookahead(self):
        """Test if lookahead works without error."""
        inputs_batch = np.random.uniform(0, 1,
                                         (self.batch_num, self.input_size))
        labels_batch = np.random.uniform(0, 1, (self.batch_num, 1))

        inputs = tf.placeholder(dtype=tf.float32, shape=(1, self.input_size))
        labels = tf.placeholder(dtype=tf.float32, shape=(1, 1))

        outputs = tf.layers.dense(inputs, 1)
        loss = tf.reduce_mean(tf.abs(outputs - labels))
        fast_opt = tf.train.AdamOptimizer(learning_rate=0.001)

        global_step = tf.train.get_or_create_global_step()
        optimizer = lookahead.LookaheadOptimizer(fast_opt)
        train_step = optimizer.minimize(loss, global_step=global_step)

        with self.cached_session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            for step in range(self.batch_num):
                sess.run(
                    [global_step, loss, outputs, train_step],
                    feed_dict={
                        inputs: inputs_batch[[step], :],
                        labels: labels_batch[[step], :]
                    })

                if step % self.k == 0:
                    for weight in tf.trainable_variables():
                        self.assertAllClose(
                            weight.eval(),
                            optimizer.get_slot(weight, 'slow').eval())
                else:
                    for weight in tf.trainable_variables():
                        self.assertNotAllClose(
                            weight.eval(),
                            optimizer.get_slot(weight, 'slow').eval())


if __name__ == '__main__':
    tf.test.main()
