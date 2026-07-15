#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from scripts.fisher_forecast import fisher_forecast

# N32 Fisher forecast
result = fisher_forecast(r=0.003, tau=0.054, lmax=95, nside=32, noise_std_uK=0.0, f_sky=1.0)
print(f'N32 Fisher: sigma_r={result["sigma_r"]:.6f}, sigma_tau={result["sigma_tau"]:.6f}')
print(f'N32 Fisher: r_pct_error={result["sigma_r"]/0.003*100:.2f}%, tau_pct_error={result["sigma_tau"]/0.054*100:.2f}%')

# N16 Fisher forecast
result16 = fisher_forecast(r=0.003, tau=0.054, lmax=47, nside=16, noise_std_uK=0.0, f_sky=1.0)
print(f'N16 Fisher: sigma_r={result16["sigma_r"]:.6f}, sigma_tau={result16["sigma_tau"]:.6f}')
print(f'N16 Fisher: r_pct_error={result16["sigma_r"]/0.003*100:.2f}%, tau_pct_error={result16["sigma_tau"]/0.054*100:.2f}%')

# N128 Fisher forecast
result128 = fisher_forecast(r=0.003, tau=0.054, lmax=383, nside=128, noise_std_uK=0.0, f_sky=1.0)
print(f'N128 Fisher: sigma_r={result128["sigma_r"]:.6f}, sigma_tau={result128["sigma_tau"]:.6f}')
print(f'N128 Fisher: r_pct_error={result128["sigma_r"]/0.003*100:.2f}%, tau_pct_error={result128["sigma_tau"]/0.054*100:.2f}%')