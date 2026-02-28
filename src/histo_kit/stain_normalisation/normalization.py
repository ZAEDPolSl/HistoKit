import staintools
import numpy as np
np.bool = bool
# Read data
target = staintools.read_image("/mnt/warehouse/Projects/HE/GrandQC/GrandQCDataset/MPP10/Colon/f93aef69-f3ba-d018-c3df-4d10344df03f_145423/ALL/f93aef69-f3ba-d018-c3df-4d10344df03f_145423 [d=3.79801,x=35002,y=64171,w=1945,h=1945].jpg")
to_transform = staintools.read_image("/mnt/warehouse/Projects/HE/GrandQC/GrandQCDataset/MPP10/Colon/0dc0a59e-eed8-a877-89ca-f550e6af1913_165712/ALL/0dc0a59e-eed8-a877-89ca-f550e6af1913_165712 [d=3.79801,x=38892,y=36947,w=1944,h=1945].jpg")

# Standardize brightness (optional, can improve the tissue mask calculation)
target = staintools.LuminosityStandardizer.standardize(target)
to_transform = staintools.LuminosityStandardizer.standardize(to_transform)



# Stain normalize
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target)
transformed = normalizer.transform(to_transform)
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(target)
plt.axis("off")
plt.title("Transformed image")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(to_transform)
plt.axis("off")
plt.title("Transformed image")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(transformed)
plt.axis("off")
plt.title("Transformed image")
plt.show()