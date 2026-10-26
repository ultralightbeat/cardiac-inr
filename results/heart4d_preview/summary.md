# 4D heart: patient051

source: field; grid 104x104x61 (voxel 1.64x1.64x1.64 mm), 30 time points; blood threshold -0.247

| method | LV EDV | LV ESV | LV EF | RV EDV | RV ESV | RV EF |
|---|---|---|---|---|---|---|
| Expert labels (ED, ES) | 212 | 120 | 44% | 142 | 67 | 53% |
| Threshold, scanner slices | 176 | 93 | 47% | 129 | 61 | 53% |
| Threshold, neural field | 175 | 93 | 47% | 127 | 62 | 51% |

Volumes in ml. EF = (EDV - ESV) / EDV. Expert labels include the papillary muscles in the cavity, the intensity threshold does not, so threshold volumes are expected to be somewhat lower.
