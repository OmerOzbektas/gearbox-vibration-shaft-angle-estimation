# Can gearbox vibration provide a shaft-angle reference?

**Reconstructing idler-shaft rotation from an accelerometer signal and checking it against a real tachometer.**

A tachometer tells us where a shaft is in its rotation. An accelerometer does not. I wanted to see how far I could get using only a gear-mesh vibration component as the angular reference for **Time Synchronous Averaging (TSA)**.

In a selected **40 Hz eccentric-gear recording**, the vibration-based TSA profile matched the tachometer-based profile with **0.999997 correlation**. A constant-RPM estimate was nearly as good (**0.999995**). The more interesting part came when I expanded the comparison: tracking instantaneous phase was not always better, and a broader automatic peak search picked the wrong spectral component in one case.

![Tachometer, Hilbert and constant-RPM TSA profiles for the eccentric gearbox at nominal 40 Hz using the selected 48X carrier.](figures/tsa_eccentric_40hz.png)

*Same vibration envelope, three angular references. Eccentric Spur 3, nominal 40 Hz, high load. The Hilbert and constant-RPM references both come from vibration; the tachometer is used for comparison.*

| 0.999997 | 0.1308° | 21.16× |
|:---:|:---:|:---:|
| TSA correlation against tachometer<br>**40 Hz · eccentric · selected carrier** | Hilbert idler-angle RMS error<br>**40 Hz · eccentric · selected carrier** | Eccentric / healthy normalized envelope power near idler 1X<br>**40 Hz** |

The 21.16× value measures a modulation difference between two recordings, not fault-classification accuracy. These are results from individual high-load recordings, not averages across independent trials.

**[Read the code walkthrough →](docs/NOTEBOOK_WALKTHROUGH.md)** · **[Open the notebook →](Portfolyo2_Colab_Final.ipynb)** · **[PHM Society dataset →](https://phmsociety.org/public-data-sets/)**

## What I built

The gearbox has a 32/96 first-stage gear pair and a 48-tooth gear on the idler shaft. That puts the idler speed at one-third of input speed. A measured vibration component near **48× input speed** can therefore be interpreted as **144 cycles per idler revolution** when the intended gear-mesh harmonic is isolated.

The processing chain was:

`Gear ratios → expected mesh harmonics → vibration spectrum → narrowband Hilbert phase → idler angle → angular resampling → TSA`

I checked the estimated angle against a tachometer-derived angle. I also used the same vibration-derived average speed to make a **constant-RPM baseline**, so the comparison tests whether tracking the phase fluctuations adds anything beyond knowing the average speed.

The [notebook walkthrough](docs/NOTEBOOK_WALKTHROUGH.md) follows every code cell, including the FFT and windowing choices, tachometer interpolation, envelope analysis, phase conversion, and the unsuccessful filter tests.

## Results: angle accuracy and profile agreement are not the same thing

The main phase-error comparison uses relative **idler-shaft degrees**, aligned at the start of the evaluation window. Lower RMS is better.

| Recording | Hilbert angle RMS | Constant-RPM angle RMS | Hilbert TSA correlation | Constant-RPM TSA correlation |
|---|---:|---:|---:|---:|
| Healthy · 30 Hz | 0.0900° | **0.0828°** | 0.9999965 | 0.9999953 |
| Eccentric · 30 Hz | 0.3566° | **0.2665°** | 0.9999668 | 0.9999876 |
| Healthy · 40 Hz | **0.1637°** | 0.2422° | 0.9999839 | 0.9999647 |
| Eccentric · 40 Hz, selected 48X carrier | **0.1308°** | 0.1343° | 0.9999971 | 0.9999954 |

The first three rows come from the saved final validation sweep; the fourth comes from the earlier **targeted carrier-selection analysis** in the same notebook. This difference is important, because the final sweep did **not** select the same component for eccentric 40 Hz.

### The automatic carrier-selection failure

The last notebook section tries a common search rule across 30, 40 and 50 Hz: take the largest FFT peak within **±8 Hz of 48 × nominal input frequency**. For eccentric 40 Hz, this picks a component with **observed order 48.3359**, rather than one near 48.0000. The saved output gives **57.3750° Hilbert angle RMS** and **0.6202607 TSA correlation**. In contrast, the earlier targeted search picked **1912.5766 Hz**, producing **0.1308° RMS** and **0.9999971 correlation**.

This is not two independent recordings giving different outcomes. It is the **same operating condition analyzed with different carrier-selection rules**. The unwanted peak is separated from the targeted carrier by roughly the idler 1X frequency, which is consistent with sideband selection, although the saved results alone do not conclusively establish its source.

![Saved four-panel final sweep. The eccentric 40 Hz panel visibly fails to align because a different carrier was selected.](figures/final_validation_grid.png)

*The final sweep is included as it was produced, including the failed eccentric 40 Hz panel. I have not replaced its results with the successful targeted run.*

## What the faulted gear looked like in the vibration signal

Before estimating shaft angle, I compared healthy Spur 1 with eccentric Spur 3 under the same nominal 40 Hz and high-load setting. Eccentricity changes the way the teeth engage around an idler revolution, so I examined modulation around the mesh harmonic rather than judging the raw FFT by its largest peaks.

![Healthy and eccentric normalized-envelope spectra at 40 Hz.](figures/envelope_spectrum_40hz.png)

The normalized envelope power near idler 1X was **0.014565** for healthy and **0.308247** for eccentric: **21.16×** in this recording pair. Splitting each tachometer-synchronized TSA into two halves gave profile correlations of **0.9448** (healthy) and **0.9937** (eccentric). These checks concern within-recording modulation and repeatability, not a tested fault classifier.

## Why I also tested constant RPM

I originally wanted the Hilbert phase to track the short-term rotation more closely than a straight line fitted to average speed. In the **30 Hz eccentric recording**, it did not:

| Phase-filter half-width | Hilbert idler-angle RMS error |
|---|---:|
| ±3 Hz | **0.3566°** |
| ±12 Hz | 6.2538° |
| ±20 Hz | 9.3981° |

![Spectrum of the tachometer deviation and Hilbert correction for eccentric 30 Hz.](figures/deviation_spectrum_eccentric_30hz.png)

The tachometer-derived angular deviation had a prominent component around **10 Hz**. The narrowband Hilbert correction's largest peak in the inspected range was around **6 Hz**, and their correlation was **0.0478**. Widening the phase filter made the estimate worse, not better. A method that reproduces the **averaged** angle-synchronized profile is not necessarily following every **instantaneous** fluctuation accurately.

## Files and how to run it

| File | Contents |
|---|---|
| [`Portfolyo2_Colab_Final.ipynb`](Portfolyo2_Colab_Final.ipynb) | Original 53-cell Python notebook, with saved plots and outputs |
| [`docs/NOTEBOOK_WALKTHROUGH.md`](docs/NOTEBOOK_WALKTHROUGH.md) | Equations, actual code excerpts, parameter choices and interpretation, in notebook order |
| [`figures/`](figures/) | Figures extracted from the notebook's saved outputs |
| [`requirements.txt`](requirements.txt) | Python dependencies |

The data comes from the **[PHM Society 2009 Gearbox Fault Diagnosis dataset](https://phmsociety.org/public-data-sets/)** ([apparatus description](https://phmsociety.org/data-analysis-competition/apparatus/)). I used the input-side accelerometer, plus the input-shaft tachometer as a reference. The records have three synchronized voltage channels sampled at **66,666.67 Hz**, with **10 tachometer pulses per input revolution**. The code expects the labeled `.txt` files in the same working directory as the notebook, for example `spur 1_40hz_High_1.txt`.

Install the Python dependencies with `pip install -r requirements.txt`, obtain the labeled dataset from PHM Society, and run the notebook from top to bottom. The raw recordings are **not included** here. Its final loop requests **50 Hz** files too, but the saved run reports both 50 Hz recordings as **missing**. There are no 50 Hz measurements to report from that loop.

## Boundaries of this project

These tests use high-load recordings from the healthy and eccentric spur configurations, principally at **30 and 40 Hz**. They examine a shaft-angle estimate that assumes known gear ratios and approximate speed. They do not establish performance during speed ramps, generalization across repeated acquisitions, or real-time operation; `sosfiltfilt` is an offline filter. The recovered angle is **relative angle**, not an independently measured absolute shaft-zero position. The tachometer is still needed to assess error, even when its samples are not used to form the vibration-based estimate.

I see this as a practical signal-processing exercise in using machine geometry to constrain what I look for in noisy measurements—and in checking a promising result against a simpler baseline and a failure case.
