# Compatibility of MMClassification 0.x

## MMClassification 0.20.1

### MMCV compatibility

In Twins backbone, we use the `PatchEmbed` module of MMCV, and this module is added after MMCV 1.4.2.
Therefore, we need to update the mmcv version to 1.4.2.
