"""Render one structural pose; this is direct posing, not dynamic motion."""

import math
import os
import struct
import zlib
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]


def png(path, pixels):
    height, width, channels = pixels.shape
    assert channels == 3
    raw = b"".join(b"\0" + pixels[row].tobytes() for row in range(height))
    def chunk(kind, data):
        return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data)&0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">2I5B",width,height,8,2,0,0,0))
                     +chunk(b"IDAT",zlib.compress(raw,6))+chunk(b"IEND",b""))


m=mujoco.MjModel.from_xml_path(str(ROOT/"models"/"scara.xml"))
d=mujoco.MjData(m)
d.qpos[:]=[math.radians(35),math.radians(-65),.055,math.radians(20),.014,.014]
mujoco.mj_forward(m,d)
with mujoco.Renderer(m,height=720,width=960) as renderer:
    renderer.update_scene(d,camera="presentation")
    png(ROOT/"figures"/"model_preview.png",renderer.render())
print("Saved figures/model_preview.png (direct structural pose).")
