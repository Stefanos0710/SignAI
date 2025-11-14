from pathlib import Path

BASE_DIR = Path("app") / "dist"
TARGET_DIRS = ["SignAI - Desktop", "SignAI - Updater"]
TARGET_EXE_NAMES = ["SignAI - Desktop.exe", "SignAI - Updater.exe"]

MANIFEST_TEMPLATE = """
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel 
          level="requireAdministrator"
          uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
</assembly>
"""


def create_admin_manifests():
    created = []
    for folder_name, exe_name in zip(TARGET_DIRS, TARGET_EXE_NAMES):
        folder = BASE_DIR / folder_name
        exe_path = folder / exe_name
        manifest_path = exe_path.with_suffix(exe_path.suffix + ".manifest")

        folder.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(MANIFEST_TEMPLATE, encoding="utf-8")
        created.append(manifest_path)
    return created


if __name__ == "__main__":
    manifests = create_admin_manifests()
    for m in manifests:
        print(f"Manifest erstellt: {m}")
