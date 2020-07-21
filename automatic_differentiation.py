# Tutorial for advanced Automactic Differentiation
# Original link: https://www.tensorflow.org/guide/advanced_autodiff?authuser=1

# imports
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8,6)

# Controlling gradient recording
# If you wish to stop recording gradients, you can use GradientTape.stop_recording() to temporarily suspend recording.

x = tf.Variable(2.0)
y = tf.Variable(3.0)
with tf.GradientTape() as t:
    x_sq = x * x
    with t.stop_recording():
        y_sq = y * y
    z = x_sq + y_sq
grad = t.gradient(z, {'x': x, 'y': y})
print('dz/dx:', grad['x'])
print('dz/dy:', grad['y'])

# If you wish to start over entirely, use reset().
reset = True
with tf.GradientTape() as t:
    y_sq = y * y
    if reset:
        # throw out all the tape recorded so far
        t.reset()
    z = x * x + y_sq
grad = t.gradient(z, {'x': x, 'y': y})
print('reset case: dz/dx:', grad['x'])
print('reset case: dz/dy:', grad['y'])

# Stop gradient
with tf.GradientTape() as t:
    y_sq = y * y
    z = x**2 + tf.stop_gradient(y_sq)
grad = t.gradient(z, {'x': x, 'y': y})
print('stop case dz/dx:', grad['x'])
print('stop case dz/dy:', grad['y'])

# Custom gradients
# In some cases, you may want to control exactly how gradients are calculated rather than using the default. These situations include:
# 
# There is no defined gradient for a new op you are writing.
# The default calculations are numerically unstable.
# You wish to cache an expensive computation from the forward pass.
# You want to modify a value (for example using: tf.clip_by_value, tf.math.round) without modifying the gradient.
# For writing a new op, you can use tf.RegisterGradient to set up your own. See that page for details. (Note that the gradient registry is global, so change it with caution.)
# 
# For the latter three cases, you can use tf.custom_gradient.
# 
# Here is an example that applies tf.clip_by_norm to the intermediate gradient.
# Establish an identity operation, but clip during the gradient pass
@tf.custom_gradient
def clip_gradients(y):
  def backward(dy):
    return tf.clip_by_norm(dy, 0.5)
  return y, backward

v = tf.Variable(2.0)
with tf.GradientTape() as t:
  output = clip_gradients(v * v)
print(t.gradient(output, v))  # calls "backward", which clips 4 to 2

# Multiple tapes
# Multiple tapes interact seamlessly. For example, here each tape watches a different set of tensors:
x0 = tf.constant(0.0)
x1 = tf.constant(0.0)

with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
    tape0.watch(x0)
    tape1.watch(x1)
    y0 = tf.math.sin(x0)
    y1 = tf.math.sigmoid(x1)
    y = y0 + y1
    ys = tf.reduce_sum(y)
print('multiple tapes case: dys/dx0:', tape0.gradient(ys,x0).numpy())
print('multiple tapes case: dys/dx1:', tape1.gradient(ys,x1).numpy())

# Higher-order gradients
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t2:
  with tf.GradientTape() as t1:
    y = x * x * x

  # Compute the gradient inside the outer `t2` context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t1.gradient(y, x)
d2y_dx2 = t2.gradient(dy_dx, x)

print('dy_dx:', dy_dx.numpy())  # 3 * x**2 => 3.0
print('d2y_dx2:', d2y_dx2.numpy())  # 6 * x => 6.0

# Example: Input gradient regularization

# Below is a naive implementation of input gradient regularization. The implementation is:
# 
# Calculate the gradient of the output with respect to the input using an inner tape.
# Calculate the magnitude of that input gradient.
# Calculate the gradient of that magnitude with respect to the model.
x = tf.random.normal([7,5])
layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)

with tf.GradientTape() as t2:
    # The inner tape only takes the gradient with respect to the input
    # not the variables.
    with tf.GradientTape(watch_accessed_variables=False) as t1:
        t1.watch(x)
        y = layer(x)
        out = tf.reduce_sum(layer(x)**2)
    # 1. Calculate the input gradient
    g1 = t1.gradient(out,x)
    # 2. Calculate the magnitude of the input gradient
    g1_mag = tf.norm(g1)
# 3. Calculate the gradient of the magnitude with respect to the model.
dg1_mag = t2.gradient(g1_mag, layer.trainable_variables)
for var in dg1_mag:
    print('var shape:', var.shape)

# Jacobians
# The GradientTape.jacobian method allows you to efficiently calculate a Jacobian matrix.
# Scalar source
x = tf.linspace(-10.0, 10.0, 200+1)
delta = tf.Variable(0.0)

with tf.GradientTape() as tape:
    y = tf.nn.sigmoid(x+delta)
dy_dx = tape.jacobian(y, delta)
print('y.shape:', y.shape)
print('dy_dx:', dy_dx.shape)

# Tensor source
x = tf.random.normal([7,5])
layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)
with tf.GradientTape(persistent=True) as tape:
    y = layer(x)
print('y shape:', y.shape)
print('layer.kernel.shape:', layer.kernel.shape)
j = tape.jacobian(y, layer.kernel)
print('j.shape:', j.shape)


