import tensorflow as tf

print("Versão do TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\n✅ Sucesso! O TensorFlow está usando a GPU: {gpus}")
else:
    print("\n❌ A GPU ainda não foi detectada. Verifique os logs do terminal acima.")

