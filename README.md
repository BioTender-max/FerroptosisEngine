# FerroptosisEngine

**Iron-Dependent Cell Death Analysis Pipeline**

A pure-Python computational engine for ferroptosis analysis integrating transcriptomics, lipidomics, and iron metabolism.

## Features
- Ferroptosis sensitivity scoring (GPX4/SLC7A11/ACSL4/FSP1 axis)
- Lipid peroxidation cascade ODE simulation (PUFA → PUFA-OOH → MDA)
- Iron metabolism network ODE (LIP, ferritin, transferrin, Fe2+)
- RSL3 (GPX4 inhibitor) and Erastin (xCT inhibitor) dose-response modeling
- Ferroptosis vs apoptosis vs necroptosis vs survival classifier (nearest centroid)
- Cell death trajectory analysis

## Results
- 200 cell lines, 5000 genes, 150 lipid species
- Ferroptosis score r=0.843 vs true sensitivity
- MDA ratio (sensitive/resistant): 5.3x
- RSL3/Erastin AUC r=0.968
- Cell death classifier accuracy=0.997

## Usage
```bash
pip install numpy scipy matplotlib
python ferroptosis_engine.py
```

## Tags
`ferroptosis` `cell-death` `apoptosis` `gpx4` `lipid-peroxidation` `iron-metabolism`
