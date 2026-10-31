# Repository distribution checks

Checked 31 October 2026 against scientific records through 30 October 2026.

- Editable package installation succeeded in an isolated environment using the existing Python 3.9 scientific runtime and setuptools 69.0.3.
- The 74 inherited model, input, NLO-interface, derivative, and calibration tests passed.
- Four additional public-tool tests passed for PDF conventions, correlated whitening and derivatives, covariance rejection, and solver-cache equality/invalidation.
- The forward quickstart found all 102 requested coordinate pairs and returned finite real/imaginary predictions for 160 observations from a fresh random model.
- The artificial-record calibration example completed.
- All 598 imported source/artifact hashes matched the distribution manifest.
- All six original root scripts matched the previous GitHub version byte for byte.
- New documentation's relative file links resolved; core Python sources parsed successfully.

The inherited derivative test emits one PyTorch warning when converting a differentiable scalar to a float for assertion. The numerical check passes. No source scientific implementation was changed to silence it.

These checks establish packaging and interface behavior. No inverse fit, native evolution rebuild, new statistical campaign, cluster submission, or independent replication of all paper results was performed for this repository update. Native and archived workflow sources retain their original runtime requirements.
