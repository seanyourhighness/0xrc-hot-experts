# Published qualification evidence

Benchmark records in this directory separate measured facts from recommendations. They are intended
to make results reproducible and falsifiable, not to imply that tokens per second transfer across
different hardware, prompts, samplers, models, or runtime tags.

## Evidence levels

- **Historical Champion evidence:** produced by the exact validated four-file Champion runtime
  snapshot before the installer/tuner wrapper was packaged. Runtime base commit and patch digest are
  recorded. These rows establish the behavior that the public workflow must reproduce.
- **Champion Runtime receipt:** produced end-to-end by `doctor → tune → verify` from an immutable
  public release tag. This is the target evidence level for new systems.
- **Pending:** hardware is named, but no performance or correctness claim is made.

## Systems

| Record | Runtime path | Result |
|---|---|---:|
| [RTX 5090 / Ryzen / DDR5](rtx5090-ryzen-ddr5-2026-09-20.json) | profile-seeded adaptive, MTP3 | 64.04 tok/s |
| [RTX 4090 / Z840 / DDR4 ECC](rtx4090-z840-ddr4-ecc-2026-09-22.json) | static profile, MTP3 | 26.53 tok/s mean |
| [RTX 4070 Ti / 48 GB template](rtx4070ti-48gb-template.json) | not yet qualified | pending |

The 5090 used categorical sampling at temperature 0.6; the 4090 controlled sweep used temperature
0. Their throughput values therefore must not be treated as a direct GPU comparison.

## Adding a machine

1. Install an immutable release tag in a clean environment.
2. Run `doctor`, then tune with `--no-mtp` first.
3. Run `verify` without either safety override.
4. Preserve `doctor.json`, `champion-profile.json`, and
   `champion-profile.verification.json` outside the Git repository if they expose local paths.
5. Copy the template, remove private paths, include SHA-256 digests for the three receipts, and add
   only values present in those receipts.
6. If desired, run a second complete qualification with MTP enabled and publish it as a separate
   record.

Never infer missing metrics, combine best values from different trials into one row, or call an
adaptive arm qualified when the sealed profile selected static placement.
