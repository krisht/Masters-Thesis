def train_model(outdir=None):
     loss = triplet_loss(alpha=alpha)
     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
     optim = optimizer.minimize(loss=loss)
     sess.run(tf.global_variables_initializer())

     count = 0
     ii = 0
     val_percentage = 0
     val_conf_matrix = 0
     epoch = -1
     while True:
          epoch += 1
          ii = 0
          count = 0
          temp_count = 0
          full_loss = 0
          while ii <= batch_size:
               ii += 1
               a, p, n = get_triplets()

               temploss = sess.run(loss, feed_dict={anchor: a, positive: p, negative: n})

               if temploss == 0:
                    ii -= 1
                    count += 1
                    temp_count += 1
                    continue

               full_loss += temploss

               if ((ii + epoch * batch_size) % 1000 == 0):
                    loss_mem_skip.append(full_loss / (1000.0 + temp_count))
                    loss_mem.append(full_loss / (1000.0))
                    full_loss = 0
                    temp_count = 0
                    get_loss(loss_mem, loss_mem_skip)

               _, a, p, n = sess.run([optim, anchor_out, positive_out, negative_out], feed_dict={anchor: a, positive: p, negative: n})

               d1 = np.linalg.norm(p - a)
               d2 = np.linalg.norm(n - a)

               if DEBUG:
                    print("Epoch: %2d, Iter: %7d, IterSkip: %7d, Loss: %.4f, P_Diff: %.4f, N_diff: %.4f" % (epoch, ii, count, temploss, d1, d2))
          val_percentage, val_conf_matrix = validate(epoch)
     sess.close()
     return epoch, val_percentage, val_conf_matrix
