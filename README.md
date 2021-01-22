# MeerKAT_LightCurve_Analysis
Code for the investigation and analysis of light curves in MeerKAT observations of GX 339-4. This work is part of my PhD thesis at the University of Manchester.

The Python (.py) and Jupyter Notebook (.ipynb) files with the same name contain the same code. The Jupyter Notebooks have additional notes and descriptions.

### The notebook and scripts

<ul>
  <li><em>TraP_SourceProcessing.ipynb</em> (or <em>TraP_SourceProcessing.py</em>) contains the code for processing information from LOFAR TraP databases. It makes an individual CSV file (in a nice Pandas format) for each source that TraP detected</li>
  <li><em>CorrelationInvestigation.ipynb</em> (or <em>CorrelationInvestigation.py</em>) uses the outputs of <em>TraP_SourceProcessing.ipynb</em> and uses the Pearson's r correlation coefficient to produce files with the correlations of all sources in the light curve files</li>  
  <li><em>EpochScaling.ipynb</em> uses outputs from <em>TraP_SourceProcessing.ipynb</em> to model systematics in the light curves of sources detected and tracked by TraP</li>
  <li><em>CorrelationInvestigation_Plots.ipynb</em> (or <em>CorrelationInvestigation_Plots.py</em>) uses the outputs of <em>CorrelationInvestigation.ipynb</em>, <em>TraP_SourceProcessing.ipynb</em> and <em>EpochScaling.ipynb</em> to plot a variety of plots to investigate the light curves and properties of the field</li>  
</ul>

This code has a DOI to help you reference it quickly and easily. If you use the code in this repo, please make sure you cite it correctly!

<a href="https://zenodo.org/badge/latestdoi/331851041"><img src="https://zenodo.org/badge/331851041.svg" alt="DOI"></a>

I am part of the MeerTRAP team, based at the University of Manchester and funded by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 694745).
