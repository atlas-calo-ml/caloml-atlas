The "main" environment is contained in `ml4p.yml`. Currently using TF 2.2 (until whenever conda finally adds TF 2.5...)

We also have a "dev" environment in `ml4p2.yml`. Currently using TF 2.3. This has TF installed via pip within conda, which is kind of hacky but necessary as conda's TF 2.4 has issues with GPU detection, and tensorflow-gpu 2.3 doesn't seem available for non-Windows. Also has AutoKeras (which require TF 2.3.0+).
