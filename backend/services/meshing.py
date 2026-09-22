"""
Stage 09.1 - surface mesh from the dense cloud (ROADMAP.md P9.1, stretch goal).

Screened Poisson reconstruction (Open3D) turns the dense colored cloud into a watertight-ish triangle
mesh with vertex colors, then trims the low-density "balloon" artifacts Poisson creates outside the
observed region. Exports OBJ (+ vertex colors) and GLB.

This is a stretch feature: it can be unstable on sparse/low-overlap clouds, so callers should treat a
RuntimeError as "mesh unavailable" and fall back to the dense point cloud.
"""
import os
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d


def build_mesh(stage_dir: str, out_obj: Optional[str] = None, out_glb: Optional[str] = None,
               poisson_depth: int = 9, density_quantile: float = 0.04,
               target_points: int = 80000) -> Dict[str, Any]:
    ply = os.path.join(stage_dir, "dense.ply")
    if not os.path.isfile(ply):
        raise FileNotFoundError("Build the dense cloud first.")
    pcd = o3d.io.read_point_cloud(ply)
    if len(pcd.points) < 500:
        raise RuntimeError("Too few dense points to mesh.")

    # Downsample for stable, fast normals + Poisson.
    if len(pcd.points) > target_points:
        vox = float(np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))) / 400.0
        if vox > 0:
            pcd = pcd.voxel_down_sample(vox)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    try:
        pcd.orient_normals_consistent_tangent_plane(12)
    except Exception:
        pcd.orient_normals_towards_camera_location(pcd.get_center())

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    if len(mesh.triangles) == 0:
        raise RuntimeError("Poisson reconstruction produced no surface.")

    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))  # trim balloon artifacts
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise RuntimeError("Mesh empty after cleanup.")
    mesh.compute_vertex_normals()

    out_obj = out_obj or os.path.join(stage_dir, "mesh.obj")
    out_glb = out_glb or os.path.join(stage_dir, "mesh.glb")
    o3d.io.write_triangle_mesh(out_obj, mesh, write_vertex_colors=True)

    glb_ok = False
    try:                                    # GLB via trimesh (handles vertex colors cleanly)
        import trimesh
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        vc = np.asarray(mesh.vertex_colors)
        colors = (np.clip(vc, 0, 1) * 255).astype(np.uint8) if len(vc) == len(v) else None
        tm = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors, process=False)
        tm.export(out_glb)
        glb_ok = True
    except Exception as e:
        print(f"[mesh] GLB export skipped: {e}")

    return {
        "available": True,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "obj": out_obj,
        "glb": out_glb if glb_ok else None,
        "poisson_depth": poisson_depth,
    }
