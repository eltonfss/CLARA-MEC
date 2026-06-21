import tensorflow as tf
from mec.environment import MECEnvironment
from models.multitask_policy_model import MultiTaskPolicyModel


def run_mtda_experiment(config, dataset):

    model_wrapper = MultiTaskPolicyModel(config)
    model = model_wrapper.get_model()

    (x_train, y_train), (x_test, y_test) = dataset

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["training"]["learning_rate"]
    )

    env = MECEnvironment(config)

    epochs = config["federated_learning"]["local_epochs"]

    for epoch in range(epochs):

        for x_batch, y_batch in x_train:

            with tf.GradientTape() as tape:

                off_logits, res_logits = model(x_batch, training=True)

                y_off = y_batch["offload_decision"]
                y_res = y_batch["resource_allocation"]

                loss_off = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        y_off,
                        off_logits,
                        from_logits=True
                    )
                )

                loss_res = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        y_res,
                        res_logits,
                        from_logits=True
                    )
                )

                loss = loss_off + loss_res

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

            d_t = tf.cast(tf.round(tf.sigmoid(off_logits)), tf.int32)
            r_t = tf.argmax(res_logits, axis=1)

            for decision, resource in zip(d_t, r_t):

                decision = int(decision.numpy())
                resource = int(resource.numpy())

                env.step((decision, resource))

    return model