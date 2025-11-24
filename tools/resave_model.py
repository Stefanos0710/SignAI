"""
resave_model.py

Lädt ein Keras-Modell und speichert es neu. Enthält optionalen "shim"-Trick für problematische Klassen wie
keras.src.ops.numpy.Any, indem beim Laden ein minimaler Ersatz (Ignore-Name) registriert wird.
Usage:
  python tools\\resave_model.py PATH_TO_MODEL [OUT_PATH]

Wenn OUT_PATH fehlt, wird PATH_TO_MODEL.resaved.keras erstellt.
"""
import sys, os, traceback


def AnyShimClassFactory():
    import tensorflow as _tf
    class AnyShim:
        def __init__(self, axis=-1, keepdims=False, **kwargs):
            # Accept and ignore 'name' and other unexpected kwargs
            if 'name' in kwargs:
                kwargs.pop('name')
            self.axis = axis
            self.keepdims = keepdims
        def get_config(self):
            return {'axis': self.axis, 'keepdims': self.keepdims}
        @classmethod
        def from_config(cls, config):
            return cls(**config)
        def __call__(self, inputs, **kwargs):
            # Perform boolean any reduction similar to keras.ops.numpy.Any
            try:
                x_bool = _tf.cast(inputs, _tf.bool)
                return _tf.reduce_any(x_bool, axis=self.axis, keepdims=self.keepdims)
            except Exception:
                # Fallback: return inputs unmodified if reduction fails
                return inputs
    return AnyShim


def main():
    if len(sys.argv) < 2:
        print('Usage: python tools\\resave_model.py PATH_TO_MODEL [OUT_PATH]')
        sys.exit(2)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path + '.resaved.keras'

    try:
        import tensorflow as tf
        from tensorflow import keras
        print('TF', tf.__version__)
    except Exception as e:
        print('TensorFlow import failed:', e)
        traceback.print_exc()
        sys.exit(1)

    if not os.path.exists(path):
        print('Input model not found:', path); sys.exit(1)

    # Attempt to load with shim for Any
    try:
        AnyShim = AnyShimClassFactory()
        custom = {
            'Any': AnyShim,
            'keras.src.ops.numpy.Any': AnyShim,
            'keras.ops.numpy.Any': AnyShim,
        }
        print('Trying load_model with custom_objects Shim...')
        m = keras.models.load_model(path, custom_objects=custom)
    except Exception as e:
        print('Initial load failed:', e)
        traceback.print_exc()
        print('Retrying without custom_objects...')
        try:
            m = keras.models.load_model(path)
        except Exception as e2:
            print('Load failed again:', e2)
            traceback.print_exc()
            sys.exit(1)

    try:
        print('Model loaded:', type(m))
        print('Saving to', out)
        m.save(out)
        print('Saved OK')
    except Exception as e:
        print('Save failed:', e)
        traceback.print_exc()

if __name__ == '__main__':
    main()
