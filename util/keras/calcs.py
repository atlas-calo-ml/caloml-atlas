import tensorflow as tf
import numpy as np
# v1 and v2 are (batches) of 4-vectors.
#@tf.function
def LorentzDot(v1,v2):
    metric = tf.constant([-1.,-1.,-1.,1.],dtype=v1.dtype)
    dot = tf.math.reduce_sum(tf.math.multiply(tf.math.multiply(v1,v2),metric),axis=-1)
    return dot

#@tf.function
def LorentzOp(vectors):
    # vectors is of shape (batch_size, n, 4)
    b = tf.shape(vectors)[0] # batch size (not necessarily known during network graph construction)
    n = vectors.shape[1] # number of vectors
    m = int((n * (n+1))/2) # number of unique dot products we can make
    
    vals = tf.Variable(tf.zeros((b,m),dtype=vectors.dtype)) # lambda:
        
    counter = 0
    for i in range(n):
        for j in range(i+1):
            vals[:,counter].assign(LorentzDot(vectors[:,i,:],vectors[:,j,:]))
            counter += 1     
    return tf.convert_to_tensor(vals)