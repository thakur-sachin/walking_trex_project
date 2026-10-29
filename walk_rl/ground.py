"""Ground contact material, shared by both stacks (pure USD authoring, call before the
simulation starts).

sim.ground_compliance in walk_task.yaml: PhysX compliant contact on the ground plane.
stiffness 0 (default) leaves the ground exactly as before: no material bound, PhysX's
default material (friction 0.5, rigid contact). A stiffness > 0 binds a material with the
same friction and that contact spring, to train or evaluate on a softer floor or sole
(MuJoCo sim2sim showed v8d needs a near-rigid floor). Rough scale for this 1.78 kg robot:
k ~ m / tc^2, so MuJoCo's default 20 ms floor is ~4.5e3 N/m and a 10 ms one ~1.8e4 N/m.
"""
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade


def apply_ground_compliance(stage: Usd.Stage, ground_path: str, gc) -> bool:
    """Bind a compliant contact material to every collider under `ground_path`.
    `gc`: namespace with stiffness_n_per_m, damping_n_s_per_m (None or stiffness 0: no-op)."""
    k = float(getattr(gc, "stiffness_n_per_m", 0.0) or 0.0) if gc is not None else 0.0
    if k <= 0.0:
        return False
    mat_path = f"{ground_path}/CompliantContactMaterial"
    mat = UsdShade.Material.Define(stage, mat_path)
    prim = mat.GetPrim()
    phys = UsdPhysics.MaterialAPI.Apply(prim)
    phys.CreateStaticFrictionAttr().Set(0.5)  # PhysX default material, as without this call
    phys.CreateDynamicFrictionAttr().Set(0.5)
    phys.CreateRestitutionAttr().Set(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    px.CreateCompliantContactStiffnessAttr().Set(k)
    px.CreateCompliantContactDampingAttr().Set(float(getattr(gc, "damping_n_s_per_m", 0.0)))
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(ground_path)):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            n += 1
    assert n > 0, f"no collider under {ground_path}"
    print(f"[ground] compliant contact: stiffness {k:g} N/m, damping {px.GetCompliantContactDampingAttr().Get():g} N*s/m "
          f"on {n} collider(s)")
    return True
