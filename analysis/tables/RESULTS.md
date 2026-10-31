# Recomputed analysis tables

CFF and response arithmetic recomputed from saved arrays. Wider-chart field errors and P2B diagnostics read from bound score records; checkpoint replay is a separate command.

## Closure errors (%)

| case | field_error_percent | ERBL_error_percent | DGLAP_error_percent | fit_Re_error_percent | fit_Im_error_percent | holdout_Re_error_percent | holdout_Im_error_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ideal_L0_Z1_s31 | 4.89668 | 5.41309 | 4.78678 | 0.92740 | 1.60140 | 0.74706 | 2.74769 |
| exp_Re_exact_Z1_s31 | 0.75017 | 0.68165 | 0.76313 | 0.00563 | 0.72860 | 0.53403 | 1.02366 |
| exp_Re_exact_Z1_s43 | 3.05123 | 2.74916 | 3.10809 | 0.00701 | 0.66855 | 0.50625 | 3.05384 |
| exp_Re_noisy_Z1_s31 | 32.69174 | 28.98518 | 33.38330 | 5.80169 | 105.87324 | 22.87846 | 156.31517 |
| exp_Re_noisy_Z1_s43 | 49.20887 | 37.04347 | 51.29569 | 7.95235 | 72.98663 | 43.92979 | 266.72708 |

## Experimental CFF residuals

| case | additional_iterations | own_RMS | local_RMS |
| --- | --- | --- | --- |
| local_cff |  | 2.43350 | 2.43350 |
| ensemble_mean | 20 | 1.32583 | 2.17274 |
| replica_000 | 20 | 1.52782 | 2.19179 |
| replica_001 | 20 | 1.58418 | 2.12619 |

## Forward information errors (%)

| noise | fit | observed_field_error_percent | A_error_percent | B_error_percent | wider_field_error_percent |
| --- | --- | --- | --- | --- | --- |
| exact | CFF | 0.92029 | 0.48477 | 0.86910 | 0.75017 |
| exact | CFF+PDF | 0.46948 | 0.67673 | 0.25165 | 0.53526 |
| half | CFF | 11.46345 | 4.98962 | 1.28139 | 28.61272 |
| half | CFF+PDF | 10.13364 | 3.54036 | 3.23636 | 38.13437 |
