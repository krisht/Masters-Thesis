def distance_metric(a, b, metric='cosine'):
     if metric == 'cosine':
          num = tf.reduce_sum(a*b, 1)
          denom = tf.sqrt(tf.reduce_sum(a*a,1))*tf.sqrt(tf.reduce_sum(b*b, 1))
          result = 1 - (num/denom)
          return result
     elif metric=='euclidean':
          return tf.reduce_sum(tf.square(tf.subtract(a, b)), 1)