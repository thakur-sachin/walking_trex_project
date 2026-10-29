"""CPU-only mesh renderer for design figures (matplotlib, no OpenGL/GPU).

The URDF meshes are in the assembly frame (each link's visual origin is minus its
zero-pose position), so loading them without the offset gives the robot at the
zero pose. Units in the files are mm.

    python redesign/render.py --out outputs/redesign/assembly.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = ROOT / "assets" / "robots" / "trex" / "meshes"
LINKS = ["base_link", "head", "tail", "left_femur", "left_tibia", "left_ankle", "left_foot",
         "right_femur", "right_tibia", "right_ankle", "right_foot"]
COLORS = {"base_link": "#c9c2b0", "head": "#e8e4da", "tail": "#e8e4da", "femur": "#4a4a4a",
          "tibia": "#5a5a5a", "ankle": "#6a6a6a", "foot": "#6d8fc7"}


def color_for(name: str) -> str:
    for k, c in COLORS.items():
        if k in name:
            return c
    return "#999999"


def load_assembly(mesh_dir: Path = MESH_DIR, links=LINKS) -> dict[str, trimesh.Trimesh]:
    return {n: trimesh.load(str(mesh_dir / f"{n}.stl")) for n in links}


def draw(ax, meshes: dict[str, trimesh.Trimesh], elev: float, azim: float, max_faces: int = 6000,
         light=(0.3, -0.5, 0.8)):
    light = np.asarray(light) / np.linalg.norm(light)
    allpts = []
    for name, m in meshes.items():
        tri = m.triangles
        nrm = m.face_normals
        if len(tri) > max_faces:  # deterministic subsample for speed (figure only)
            idx = np.linspace(0, len(tri) - 1, max_faces).astype(int)
            tri, nrm = tri[idx], nrm[idx]
        shade = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
        base = np.array(matplotlib.colors.to_rgb(color_for(name)))
        fc = np.clip(base[None, :] * shade[:, None], 0, 1)
        pc = Poly3DCollection(tri, facecolors=fc, edgecolors="none", linewidths=0)
        ax.add_collection3d(pc)
        allpts.append(m.vertices)
    p = np.vstack(allpts)
    c = (p.max(0) + p.min(0)) / 2
    r = (p.max(0) - p.min(0)).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x fwd [mm]")
    ax.set_ylabel("y left [mm]")
    ax.set_zlabel("z [mm]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "redesign" / "assembly.png"))
    ap.add_argument("--links", nargs="*", default=LINKS)
    ap.add_argument("--views", nargs="*", default=["20,-60", "0,-90", "0,0", "90,-90"])
    a = ap.parse_args()
    meshes = load_assembly(links=a.links)
    views = [tuple(map(float, v.split(","))) for v in a.views]
    fig = plt.figure(figsize=(6 * len(views), 6))
    for i, (e, az) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        draw(ax, meshes, e, az)
        ax.set_title(f"elev {e:.0f} azim {az:.0f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=80)
    print(a.out)


if __name__ == "__main__":
    main()
