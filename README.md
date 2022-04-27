# nc_AFM 

```
import ncafm
```

Author: Alyson Spitzig, Houchen Li, Talha Rehman
Licence: GNU General Public License v3



This package is meant to analyze the data acquired from non-contact Atomic Force Microscopy (nc-AFM). You could use this package to do Bayesian analysis about the parameters of your own nc-AFM measurements.

This package is mainly based on pymc3. We have several built-in models(Lennard-Jones, vdW force with different tip geometry, electrostatic forces etc.) for you to select to 


Its details functions are including :

1. the convertion between measured vibration frequency to force 
2. the force composition analyses(Lennard-Jones Force, electrostatic force and van-der-waals force etc.)
3. the determination of z_axis offset
4. and (might) also provide the analysis between different tip-sample geometry.


Please see documentation.py for further instructions.
