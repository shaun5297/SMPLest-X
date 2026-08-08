# Step 2.7A — Chest/Bust Definition Literature Review

Date: 2026-08-08
Status: `completed`
Scope: public implementations used by SMPL-Anthropometry, A2B/B2A, and the measurement code bundled with Focused Human Body Measurement.

## Outcome

The inspected public implementations expose two concrete SMPL-X chest-plane families:

1. **Bilateral nipple midpoint** — SMPL-Anthropometry and A2B use fixed-topology vertices `v3572` (`LEFT_NIPPLE`) and `v8340` (`RIGHT_NIPPLE`). The plane origin is their 3D mean; the published plane normal is the pelvis-to-spine3 joint vector. The slice is restricted to spine1/spine2 faces and its convex-hull perimeter is measured.
2. **Single surface-anchored nipple height** — the SHAPY measurement path bundled in the Focused repository defines `CW_p` from `NippleRight`, represented by `face_idx=18402` and barycentric coordinates `[0, 0, 1]`. It evaluates that anchor on the subject mesh, constructs an exactly horizontal plane at its Y coordinate, and measures a convex-hull perimeter.

No inspected code path contains a gender-conditioned chest/bust plane, a female-only landmark branch, an underbust rule, or a breast-apex search. This is a statement about the inspected public code, not a claim that ISO 8559 or every anthropometric protocol uses a sex-invariant chest/bust definition.

## Definition matrix

| Source | Landmark representation | Plane origin | Plane orientation | Region/contour handling | Sex-specific rule found? |
| --- | --- | --- | --- | --- | --- |
| SMPL-Anthropometry | `v3572`, `v8340` | bilateral 3D midpoint | pelvis-to-spine3 normal | spine1/spine2 face filter, convex hull | No |
| A2B official B2A measurer | `v3572`, `v8340` | bilateral 3D midpoint | pelvis-to-spine3 normal | spine1/spine2 face filter, convex hull | No |
| Focused repository, bundled SHAPY path | `NippleRight`, face 18402 + barycentric `[0,0,1]` | subject surface anchor | exactly horizontal `y=anchor_y` | all plane intersections, convex hull | No |

## Source-level observations

### SMPL-Anthropometry

- `measurement_definitions.py` defines SMPL-X chest circumference from `LEFT_NIPPLE`, `RIGHT_NIPPLE`, with joints `pelvis`, `spine3`.
- `landmark_definitions.py` maps those landmarks to `3572` and `8340`.
- `measure.py` averages all configured landmarks for `plane_origin`, constructs the normal from the configured joints, filters the slice to the configured body parts, then evaluates a convex-hull perimeter.
- The landmark mapping is shared by male, female, and neutral models. Gender selects the SMPL-X model, not the chest rule.

### A2B / B2A

- The official B2A measurer reproduces the same SMPL-X nipple vertex IDs and configures chest as `([pelvis, spine3], [lnipple, rnipple], [spine1, spine2])`.
- Its generic circumference implementation also uses the landmark mean, the joint-vector normal, a body-part face restriction, and a convex hull.
- No female-specific chest/bust branch was found in this implementation.

### Focused Human Body Measurement repository

- The repository states that its anthropometric output is based on ISO 8559, but the concrete code path inspected here is the bundled SHAPY mesh-intersection implementation and should be attributed as such.
- `measurement_defitions.yaml` maps `CW_p` to the single `NippleRight` anchor.
- `smplx_measurements.yaml` stores that anchor as face 18402 with barycentric coordinates `[0,0,1]`; it is therefore evaluated on each subject's deformed surface rather than reused as fixed XYZ.
- `cwh_measurements.py` builds an exactly horizontal plane at the evaluated anchor height and computes a convex-hull perimeter.
- No gender conditional or special female bust handling was found in the inspected files.

## Evidence boundary

- The findings above describe executable public definitions, not measurement accuracy; no real chest/bust ground truth is available in the current five-subject demo.
- The Focused paper/repository's broad ISO statement is not sufficient by itself to claim that the bundled `CW_p` implementation is a complete ISO-conformant female bust protocol.
- Fixed topology vertices and face+barycentric anchors both deform with subject beta. Chest therefore does **not** reproduce the Waist fixed-XYZ anchoring problem.
- SMPL-Anthropometry/A2B use an oblique joint-normal plane in the general posed case. The current project evaluates zero-pose canonical meshes with a horizontal slicer; adapting the bilateral definition to `y = mean(Y_left, Y_right)` must be reported explicitly rather than silently described as source-identical.

## Step 2.7B decision

Step 2.7B should validate plane and interval behavior before any chest argmax is implemented.

Primary candidate:

- `literature_chest_v1_candidate`: horizontal plane at the mean Y of subject-deformed `v3572/v8340`.
- Rationale: independent agreement between SMPL-Anthropometry and A2B, bilateral construction, beta-adaptive fixed topology, and direct compatibility with the frozen horizontal slicing engine.

Control candidate:

- `focused_shapy_chest_control`: horizontal plane at the subject-deformed face-18402 barycentric anchor.
- Rationale: captures the distinct single-anchor public implementation without inventing a third anatomical proxy.

For all five subjects, Step 2.7B should record:

- left/right nipple Y mismatch and midpoint Y;
- source pelvis-to-spine3 normal tilt from vertical;
- plane location relative to spine1, spine2, spine3, and shoulder/axilla levels;
- number and topology of all slice loops;
- torso-contour selection mode, containment, centroid, area, perimeter, and fallback state;
- visual front/top/oblique confirmation for both candidate planes;
- nearby raw `C(y)` samples only as diagnostics, not as an optimization rule.

Acceptance requires that the selected contour is the central chest/torso loop for all subjects, is separated from arms and shoulder/axilla topology, has no invalid/open components, and does not depend on `max(contours)` or global `argmax C(y)` assumptions.

## Sources

- SMPL-Anthropometry measurement definitions: https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/measurement_definitions.py
- SMPL-Anthropometry landmark definitions: https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/landmark_definitions.py
- SMPL-Anthropometry implementation: https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/measure.py
- A2B official B2A measurer: https://github.com/kaulquappe23/a2b_human_mesh/blob/main/anthro/measurements/measure.py
- Focused Human Body Measurement official repository: https://github.com/Eddie-cc/Focused-human-body-measurement
- Focused bundled measurement definitions: https://github.com/Eddie-cc/Focused-human-body-measurement/blob/main/data/SHAPY/mesh-mesh-intersection/data/measurement_defitions.yaml
- Focused bundled SMPL-X anchors: https://github.com/Eddie-cc/Focused-human-body-measurement/blob/main/data/SHAPY/mesh-mesh-intersection/data/smplx_measurements.yaml
- Focused bundled C/W/H implementation: https://github.com/Eddie-cc/Focused-human-body-measurement/blob/main/data/SHAPY/mesh-mesh-intersection/body_measurements/cwh_measurements.py
