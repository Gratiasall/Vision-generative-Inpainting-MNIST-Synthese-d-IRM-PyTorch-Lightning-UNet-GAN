🎯 ***Objectif du dépôt***

Ce dépôt regroupe deux Travaux Pratiques complets — chacun avec énoncé et solution — sur la vision par ordinateur générative :

**Inpainting MNIST (TP 4.1) :** restauration d’images masquées à partir d’entrées dégradées.

**Synthèse d’IRM cérébrales (TP 4.2) :** génération d’images médicales avec une architecture UNet/Generator et entraînement Lightning.

**Compétences mises en évidence :**

Conception de pipelines deep learning robustes (PyTorch + PyTorch Lightning)

Architectures UNet/Generator pour inpainting/synthèse

Prétraitement, dataloaders, masquage/augmentation

**Évaluation adaptée :** MSE (régression pixel-wise), SSIM (qualité perceptuelle)

Visualisations et suivi des entraînements, torchsummary / torchinfo, torchmetrics

📂 ***Contenu du repo***

**TP_4_1_énoncé.ipynb** — Inpainting d’images masquées (énoncé)
Problème, données, masquage, objectifs, métriques, plan d’implémentation.

**TP_4_1_ALLAKONON_VODOUMBO_solution.ipynb** — Inpainting d’images masquées (solution)
Dataset MNIST via torchvision, fonction de dégradation/masquage, modèle PyTorch, optimisation (ex. Adam), MSE comme perte/metric, visualisations avant/après.
Imports principaux : torch, torchvision, lightning, sklearn, matplotlib, torchsummary.

**TP_4_2_énoncé.ipynb** — Synthèse d’images IRM cérébrales (énoncé)
Contexte de résonance magnétique, organisation des données, consignes pour définir un Générateur/UNet, planning d’entraînement et d’évaluation.

**TP_4_2_solution.ipynb** — Synthèse d’images IRM cérébrales (solution)
Architecture UNet/Generator (blocs Down/Up), entraînement PyTorch Lightning, suivi de la SSIM (torchmetrics.image.StructuralSimilarityIndexMeasure), résumé de modèle avec torchinfo, tracés des images générées.
Imports principaux : torch, lightning, torchmetrics, matplotlib, torchinfo.

🧰 ***Stack & bibliothèques***

- Python 3, Jupyter Notebooks

- PyTorch + PyTorch Lightning (boucles d’entraînement propres et reproductibles)

- torchvision (MNIST, transforms, utilitaires)

- torchmetrics (SSIM pour la qualité perceptuelle)

- torchsummary / torchinfo (résumés d’architectures)

- scikit-learn (utilitaires), matplotlib (visualisations)

📊 ***Résultats & insights***

- Inpainting : la diminution de la MSE s’accompagne d’une réapparition nette des traits de chiffres masqués ; le ratio/placement du masque influence fortement la qualité de reconstruction.

- Synthèse IRM : l’architecture UNet/Generator améliore la SSIM en exploitant les skip connections (meilleure restitution des contours et textures fines) ; les courbes montrent une progression stable avec Lightning.