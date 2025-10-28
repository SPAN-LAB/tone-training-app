import subprocess

# Get the version from Git
try:
    version = subprocess.check_output(['git', 'describe', '--tags'], encoding='utf-8').strip()
except Exception:
    version = "1.0.0"  # Fallback version if Git fails

# Create the version info file content
version_info_content = f"""
VSVersionInfo(
    ffi=FixedFileInfo(
        filevers=({','.join(version.lstrip('v').split('.'))}, 0),
        prodvers=({','.join(version.lstrip('v').split('.'))}, 0),
        mask=0x3f,
        flags=0x0,
        OS=0x40004,
        fileType=0x1,
        subtype=0x0,
        date=(0, 0)
    ),
    kids=[
        StringFileInfo([
            StringTable(
                '040904B0', [
                    StringStruct('CompanyName', 'UW-Madison SPAN Lab'),
                    StringStruct('FileDescription', 'Plasticity Training App'),
                    StringStruct('FileVersion', '{version}'),
                    StringStruct('InternalName', 'Plasticity Training App'),
                    StringStruct('OriginalFilename', 'plasticity_training.exe'),
                    StringStruct('ProductName', 'Plasticity Training App'),
                    StringStruct('ProductVersion', '{version}')
                ])
        ]),
        VarFileInfo([VarStruct('Translation', [1033, 1200])])
    ]
)
"""

# Write the version info to a file
with open("version_info.txt", "w") as f:
    f.write(version_info_content)