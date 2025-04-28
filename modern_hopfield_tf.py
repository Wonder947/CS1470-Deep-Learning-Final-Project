# here we are going to implement the modern hopfield network in tensorflow

import tensorflow as tf

class Hopfield_tf(tf.keras.layers.Layer):
    """
    Modern Hopfield Network (single-head) with:
      - static vs. non-static (learned) K/Q projections
      - iterative updates on the associative state ξ
      - optional final output projection
    
    Future TODOs: support batch input
    
    ξ₀ = softmax(β · QKᵀ)
    ξₜ₊₁ = softmax(β · (ξₜ K) Kᵀ)
    output = ξ_final · V
    """
    def __init__(self,
                 static=True,
                 hid_dim=None,
                 scaling=1.0,
                 update_steps_max=1,
                 update_steps_eps=1e-4,
                 output_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.static = static
        if not static:
            assert hid_dim is not None, "must specify hid_dim in non-static mode"
        self.hid_dim = hid_dim
        self.scaling = scaling
        self.update_steps_max = int(update_steps_max)
        self.update_steps_eps = update_steps_eps
        self.output_dim = output_dim

    def build(self, input_shape):
        ks_shape, qs_shape, vs_shape = input_shape
        # allow K and Q to have different raw dims
        self.pattern_dim_k = ks_shape[-1]
        self.pattern_dim_q = qs_shape[-1]
        assert vs_shape[-1] == self.pattern_dim_k, "V must match K in last dim"

        if self.static:
            assert self.pattern_dim_k==self.pattern_dim_q

        if not self.static:
            # only project once, in build:
            self.k_proj = self.add_weight(
                name="k_proj",
                shape=(self.pattern_dim_k, self.hid_dim),
                initializer="glorot_uniform",
                trainable=True)
            self.q_proj = self.add_weight(
                name="q_proj",
                shape=(self.pattern_dim_q, self.hid_dim),
                initializer="glorot_uniform",
                trainable=True)

        if self.output_dim is not None:
            # V always lives in raw pattern_dim_k space → project to output_dim
            self.out_proj = self.add_weight(
                name="out_proj",
                shape=(self.pattern_dim_k, self.output_dim),
                initializer="glorot_uniform",
                trainable=True)


    def call(self, inputs):
        """
        inputs: tuple of (ks, qs, vs)
          ks: (n_keys,  pattern_dim)
          qs: (n_queries, pattern_dim)
          vs: (n_keys,  pattern_dim)
        Returns:
          output: (n_queries, output_dim or pattern_dim)
          attn_weights: (n_queries, n_keys)
        """
        ks, qs, vs = inputs

        # 1) if non-static: project into associative space
        if self.static:
            k = ks        # (n_keys,  pattern_dim)
            q = qs        # (n_queries, pattern_dim)
        else:
            k = tf.matmul(ks, self.k_proj)  # → (n_keys,  hid_dim)
            q = tf.matmul(qs, self.q_proj)  # → (n_queries, hid_dim)

        v = vs  # values remain in original space

        # 2) iterative Hopfield retrieval
        q_current = q
        xi = None

        for _ in range(self.update_steps_max):
            # compute scaled affinities
            scores = tf.matmul(q_current, k, transpose_b=True)  # (n_queries, n_keys)
            scores = scores * tf.cast(self.scaling, scores.dtype)   # apply scaling -- similar to temperature

            # softmax → association matrix
            xi_new = tf.nn.softmax(scores, axis=-1)            # (n_queries, n_keys)

            # retrieve new queries: q_new = xi_new @ k
            q_new = tf.matmul(xi_new, k)                       # (n_queries, hid_dim or pattern_dim)

            # check convergence in q-space
            diff = tf.norm(q_new - q_current, ord='euclidean')
            q_current = q_new
            xi = xi_new

            if diff < self.update_steps_eps:
                break

        # 3) final retrieval of values
        output = tf.matmul(xi, v)                             # (n_queries, pattern_dim)

        # 4) optional output projection
        if self.output_dim is not None:
            output = tf.matmul(output, self.out_proj)         # (n_queries, output_dim)

        return output