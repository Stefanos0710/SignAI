"""
fix_keras_config.py

Fixes problematic 'name' kwargs inside config.json of a .keras archive for ops like keras.src.ops.numpy.Any
by removing the 'name' key from the inner 'config' so deserialization doesn't pass an unexpected 'name' kwarg.

Usage:
  python tools\fix_keras_config.py PATH_TO_MODEL.keras [OUT_PATH.keras]

If OUT_PATH missing, will write PATH_TO_MODEL.fixed.keras
"""
import sys, os, json, zipfile, shutil, tempfile


def fix_config_dict(cfg):
    # Traverse to layers in common Keras Functional config
    changed = False
    try:
        layers = cfg.get('config', {}).get('layers', [])
    except Exception:
        layers = []
    for layer in layers:
        module = layer.get('module', '')
        class_name = layer.get('class_name', '')
        # target ops namespace
        if 'ops' in module or module.startswith('keras.src.ops'):
            # remove 'name' from inner config if present
            conf = layer.get('config', {})
            if isinstance(conf, dict) and 'name' in conf:
                conf.pop('name')
                layer['config'] = conf
                changed = True
    return cfg, changed


def main():
    if len(sys.argv) < 2:
        print('Usage: python tools\\fix_keras_config.py PATH_TO_MODEL.keras [OUT_PATH.keras]')
        sys.exit(2)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path + '.fixed.keras'

    if not os.path.exists(path):
        print('Input not found:', path); sys.exit(1)

    with zipfile.ZipFile(path, 'r') as zin:
        names = zin.namelist()
        if 'config.json' not in names and 'model_config.json' not in names:
            print('No config.json or model_config.json found in archive; nothing to fix.')
            sys.exit(0)
        cfg_name = 'config.json' if 'config.json' in names else 'model_config.json'
        cfg = json.loads(zin.read(cfg_name).decode('utf-8'))

    new_cfg, changed = fix_config_dict(cfg)
    if not changed:
        print('No changes necessary')
        # still copy original to out
        shutil.copy(path, out)
        print('Copied original to', out)
        return

    # write new archive
    tmp = out + '.tmp'
    with zipfile.ZipFile(path, 'r') as zin, zipfile.ZipFile(tmp, 'w') as zout:
        for name in zin.namelist():
            if name == cfg_name:
                zout.writestr(name, json.dumps(new_cfg))
            else:
                zout.writestr(name, zin.read(name))
    # replace/move tmp to out
    shutil.move(tmp, out)
    print('Wrote fixed model to', out)

if __name__ == '__main__':
    main()

