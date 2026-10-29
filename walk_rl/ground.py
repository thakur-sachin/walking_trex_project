"""Ground contact material, shared by both stacks (pure USD authoring, call before the
simulation starts).

sim.ground_compliance in walk_task.yaml: PhysX compliant contact on the ground plane.
stiffness 0 (default) leaves the ground exactly as before: no material bound, PhysX's
default material (friction 0.5, rigid contact). A stiffness > 0 binds a material with the
same friction and that contact spring, to train or evaluate on a softer floor or sole
(MuJoCo sim2sim showed v8d needs a near-rigid floor). Rough scale for this 1.3 kg robot:
k ~ m / tc^2, so MuJoCo's default 20 ms floor is ~4.5e3 N/m and a 10 ms one ~1.8e4 N/m.

sim.ground_friction in walk_task.yaml: the ground's static = dynamic friction (null = PhysX's default
material, 0.5, as before). PhysX averages the ground and robot friction, so with the default ground the
effective friction never goes below 0.25 whatever domain_randomization.friction_range says; a lower
ground friction lets training reach the measured floor (PLA on wood, static mu 0.27, 2026-10-21).
"""
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade


def apply_ground_compliance(stage: Usd.Stage, ground_path: str, gc, friction: float | None = None) -> bool:
    """Bind a contact material to every collider under `ground_path`.
    `gc`: namespace with stiffness_n_per_m, damping_n_s_per_m (None or stiffness 0: rigid).
    `friction`: ground static = dynamic friction (None: 0.5, PhysX's default).
    Rigid and friction None: no-op, the ground keeps PhysX's default material."""
    k = float(getattr(gc, "stiffness_n_per_m", 0.0) or 0.0) if gc is not None else 0.0
    if k <= 0.0 and friction is None:
        return False
    mu = 0.5 if friction is None else float(friction)  # 0.5: PhysX default material, as without this call
    mat_path = f"{ground_path}/{'CompliantContactMaterial' if k > 0.0 else 'GroundMaterial'}"
    mat = UsdShade.Material.Define(stage, mat_path)
    prim = mat.GetPrim()
    phys = UsdPhysics.MaterialAPI.Apply(prim)
    phys.CreateStaticFrictionAttr().Set(mu)
    phys.CreateDynamicFrictionAttr().Set(mu)
    phys.CreateRestitutionAttr().Set(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    if k > 0.0:
        px.CreateCompliantContactStiffnessAttr().Set(k)
        px.CreateCompliantContactDampingAttr().Set(float(getattr(gc, "damping_n_s_per_m", 0.0)))
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(ground_path)):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            n += 1
    assert n > 0, f"no collider under {ground_path}"
    if k > 0.0:
        print(f"[ground] compliant contact: stiffness {k:g} N/m, damping {px.GetCompliantContactDampingAttr().Get():g} N*s/m, "
              f"friction {mu:g} on {n} collider(s)")
    else:
        print(f"[ground] rigid, friction {mu:g} on {n} collider(s)")
    return True
