# GravityPy
## a python package to analyze GRAVITY data, focus on the Galactic Center science case

* What is GRAVITY? An interferometric beam combiner at the Very Large Telescope Interferometer (VLTI).
  For more information, see [here](https://www.eso.org/sci/facilities/paranal/instruments/gravity.html)
* What is the Galactic Center science case? We observe stars orbiting a supermassive black hole to learn about fundamental physics in the most extreme limits.
  For more information, see [here](https://www.mpe.mpg.de/ir/GC)

## Motivation
This package has evolved over many years of working on GRAVITY data. Here I am describing some key features of it:

## Observing preparation
We are using GravityPy to predict stellar positions and prepare observations.
The orbits used are calculated from orbital parameters in [Gillessen et al. 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...837...30G/abstract)

![image](https://github.com/widmannf/GravityPy/assets/24411509/884310e8-440d-437f-8462-5e5a684dba03)


## Fitting point an assembly of point sources
In optical interferometry, the data we observe is the Fourier transformation of the sky brightness distribution, projected onto the earth and sampled by the telescopes. The sparse sampling makes an inverse transformation impossible, and one of the ways to get the information on the sky distribution is by fitting a model directly into the Fourier plane. This is the approach taken here.

The fitting functions are the heart of this package, and many people have worked on them. The main idea is outlined in the [PhD Thesis of Idel Waisberg](https://edoc.ub.uni-muenchen.de/view/autoren/Waisberg=3AIdel_Reis=3A=3A.html). It includes treating sources with different spectral power laws and the resulting instrumental effects, such as bandwidth smearing. Additionally, it fully includes the corrections for [optical aberrations](https://ui.adsabs.harvard.edu/abs/2021A%26A...647A..59G/abstract) and can account for any distribution of point sources. A description of the mathematical concept can also be found in this [publication](https://ui.adsabs.harvard.edu/abs/2020A%26A...636L...5G/abstract).

![image](https://github.com/widmannf/GravityPy/assets/24411509/ab363352-dfb8-40ed-99ed-e9f25e9145a2)

## Graphical Interface
Many of the functionalities can also be used directly from a GUI for simpler data visualization and automated fitting.
![image](https://github.com/widmannf/GravityPy/assets/24411509/b384b8d1-fb19-4c63-bad7-e17de378bd98)
![image](https://github.com/widmannf/GravityPy/assets/24411509/a5b344a0-cd14-4386-8d0d-efdc85d67aad)
