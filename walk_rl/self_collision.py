"""PhysX self-collision for a chosen set of robot link pairs (both Isaac stacks, pxr only)."""

from __future__ import annotations

from itertools import combinations

from pxr import PhysxSchema, Usd, UsdPhysics


def apply_self_collision(stage: Usd.Stage, robot_prim_path: str, pairs: list[tuple[str, str]]) -> None:
    """PhysX self-collision between exactly `pairs` (link names): enable it on the articulation
    and filter every other pair of collision links. Joint-connected links never collide in an
    articulation anyway. Author on env_0 before cloning, like the contact reporters."""
    root = stage.GetPrimAtPath(robot_prim_path)
    arts = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(arts) == 1, f"expected one articulation root under {robot_prim_path}, found {len(arts)}"
    PhysxSchema.PhysxArticulationAPI.Apply(arts[0]).CreateEnabledSelfCollisionsAttr().Set(True)
    bodies = {p.GetName(): p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    allowed = {frozenset(pr) for pr in pairs}
    missing = {n for pr in pairs for n in pr} - set(bodies)
    assert not missing, f"self-collision pair links not found: {missing}"
    # Filter targets are each body's own collision shapes, never the body prim: a target that is
    # the articulation root (base_link) filters the whole articulation, and the links are nested
    # under it. The API sits on the non-root body of each pair.
    shapes = {n: [] for n in bodies}
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            b = p
            while not b.HasAPI(UsdPhysics.RigidBodyAPI):
                b = b.GetParent()
            shapes[b.GetName()].append(p.GetPath())
    art = arts[0].GetName()
    n_filtered = 0
    for a, b in combinations(sorted(bodies), 2):
        if frozenset((a, b)) in allowed:
            continue
        if a == art:
            a, b = b, a
        assert a != art, "the articulation root cannot carry the filter"
        rel = UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel()
        for path in shapes[b]:
            rel.AddTarget(path)
        n_filtered += 1
    print(f"[self_collision] self-collision on: {len(allowed)} link pairs collide, {n_filtered} filtered")
