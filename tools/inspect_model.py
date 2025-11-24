"""
inspect_model.py

Kurzes Hilfs-Skript zum Analysieren von Keras/TensorFlow Modelldateien (.keras/.h5/SavedModel).
Usage:
  python tools\\inspect_model.py path/to/model.keras

Gibt Keras/TensorFlow-Version, Typ, model_config (falls .keras), und sucht nach problematischen Klassen wie 'Any'.
"""
import sys, os, json, zipfile, traceback

def main():
    if len(sys.argv) < 2:
        print('Usage: python tools\\inspect_model.py PATH_TO_MODEL')
        sys.exit(2)
    path = sys.argv[1]
    print('Path:', path)

    try:
        import tensorflow as tf
        from tensorflow import keras
        print('Python:', sys.version.split()[0])
        print('TensorFlow:', tf.__version__)
        try:
            import keras as standalone_keras
            print('Standalone keras importable:', getattr(standalone_keras, '__version__', 'unknown'))
        except Exception:
            print('Standalone keras not importable')
    except Exception as e:
        print('Could not import tensorflow/keras:', e)

    if not os.path.exists(path):
        print('File not found')
        sys.exit(1)

    lower = path.lower()
    try:
        if lower.endswith('.keras') or lower.endswith('.zip'):
            print('Detected .keras (zip) format')
            with zipfile.ZipFile(path, 'r') as z:
                names = z.namelist()
                print('Archive contains:', names)
                if 'model_config.json' in names:
                    cfg = json.loads(z.read('model_config.json').decode('utf-8'))
                    print('model_config keys:', list(cfg.keys()))
                    # Pretty-print top-level config summary
                    cfg_print = json.dumps(cfg.get('config', cfg), indent=2)
                    print(cfg_print[:4000])
                    s = json.dumps(cfg)
                    if 'Any' in s or 'keras.src.ops.numpy' in s or 'Any.__init__' in s:
                        print("Found 'Any' or keras.ops.numpy references in model_config.json")
                else:
                    print('model_config.json not present in archive')
        elif lower.endswith('.h5') or lower.endswith('.hdf5'):
            import h5py
            print('Detected HDF5 (.h5) format')
            with h5py.File(path, 'r') as f:
                print('Top-level keys:', list(f.keys()))
                try:
                    print('Attrs:', dict(f.attrs))
                except Exception:
                    pass
        else:
            print('Unknown extension, attempting to read as file...')
    except Exception as e:
        print('Error during inspect:', e)
        traceback.print_exc()

if __name__ == '__main__':
    main()
