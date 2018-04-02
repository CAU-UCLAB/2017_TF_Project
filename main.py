import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

add_and_triple = adder_node * 3.

nodes = [node1, node2, node3, adder_node]
for a in nodes:
    print(a)
print(add_and_triple)


sess = tf.Session()

print(sess.run([node1, node2, node3]))
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
