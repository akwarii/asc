# Interface utilisateur pour le script d'apprentissage

Liste des arguments envisagés pour le script d'entraînement de modèles à utiliser en production. Je pense que tous les flags proposés ici sont compatibles avec le fonctionnement actuel de la libraire, sans l'avoir vérifié en détail.

Ils sont triés par catégorie pour une meilleure organisation et lisibilité. Je recommande d'utiliser des groupes d'arguments dans `argparse` pour structurer ces options de manière logique.

```python
parser.add_argument_group("[...]")
```

> **Note** : le présent fichier est un document de travail, et les arguments peuvent être ajoutés, modifiés ou supprimés au fur et à mesure de nos réflexions et de nos besoins. Certains sont inutiles ou redondants ; l'idée de ce premier jet est de proposer une gamme d'options aussi large que possible.

## Modèle

* Architecture du modèle à entraîner. (default: `painn` ?)

```bash
  --model {cegann,cegannv2,mlp,gat,painn}
```

* Hyperparamètres du modèle sous forme de paires clé=valeur (ex. k=20 hidden_channels=32). (default: `None`)

```bash
  --model-kwargs MODEL_KWARGS [MODEL_KWARGS ...]
```

> **Note** : je propose pour le moment de conserver cette structure d'arguments car il s'agit de ce qu'on fait historiquement, et elle est relativement flexible. Je reste néanmoins sceptique quant à sa facilité d'utilisation, et je pense qu'on pourrait envisager une alternative plus structurée (ex. un fichier de configuration YAML/JSON pour les hyperparamètres du modèle) si on trouve que celle-ci est trop contraignante ou sujette à erreurs.

* Se servir d'un output d'HPO pour charger les meilleurs hyperparamètres trouvés et entraîner le modèle correspondant. (default: `False`)

```bash
  --use-best-model
```

* Désactiver la compilation du modèle (`torch.compile`). (default: `False`).

```bash
  --no-compile          Disable model compilation (torch.compile). (default: False)
```

## Entraînement

* Taille de batch pour l'entraînement. (default: `128`)

```bash
  --batch-size BATCH_SIZE
```

* Rechercher automatiquement la taille de batch optimale. Remplace `--batch_size`. (default: `False`)

```bash
  --batch-size-finder
```

* Nombre d'epochs pour l'entraînement. (default: `300` ? à confirmer selon les types de modèles et de datasets)

```bash
  --epochs EPOCHS
```

* Taux d'apprentissage initial. (default: `0.006` ? à confirmer, proche de celui qu'on utilise actuellement pour PaiNN)

```bash
  --lr LR
```

* Nombre de pas ***"d'échauffement"***  pour le scheduler de taux d'apprentissage. (default: `700` ? à confirmer selon la durée typique d'un epoch)

```bash
  --warmup WARMUP
```

* il peut être intéressant de permettre du gradient clipping, comme on a observé une explosion de gradient dans certains cas

```bash
  --gradient-clip-val GRADIENT_CLIP_VAL
```

* pour les tests `PaiNN` x `custom`, on a vu que la taille des batches avait un impact important sur la stabilité de l'entraînement. Ça me paraît donc pertinent de permettre une forme d'accumulation de gradients pour compenser les contraintes de mémoire GPU et permettre l'utilisation de tailles de batch effectives plus grandes. (default: `1`)

```bash
  --accumulate-grad-batches ACCUMULATE_GRAD_BATCHES
```

## Dataset

* choix du dataset à utiliser pour l'entraînement. (default: `csg`)

```bash
  --dataset {csg,mp,aflow,gnome,custom}
```

* Proportions de données pour l'entraînement, la validation et le test. (default: `0.7`, `0.2`, `0.1`)

```bash
  --train-ratio TRAIN_RATIO
  --val-ratio VAL_RATIO
  --test-ratio TEST_RATIO
```

* Nombre de workers pour le chargement des données. (default: `8` ? Ou `5` ?)

```bash
  --num-workers NUM_WORKERS
```

* Utiliser un sampler pour les datasets déséquilibrés. (default: `True` ? ça me paraît ne pas être une mauvaise idée de l'activer par défaut)

```bash
  --use-imbalance-sampler
  --no-imbalance-sampler
```

> **Note** : je ne pense pas que les `pre-transforms` doivent être exposés en tant qu'arguments de ligne de commande, car ils me paraissent liés aux paires dataset/modèle ? Par exemple, je ne sais pas si ça a beaucoup de sens de permettre à PaiNN de fonctionner avec un LineGraph. Ça reste à discuter.

* L'écart-type pour l'augmentation par perturbation aléatoire. Si cet argument est défini, une augmentation par perturbation aléatoire sera appliquée aux données d'entraînement avec l'écart-type spécifié. Sinon, aucune augmentation par perturbation aléatoire ne sera appliquée. Il est possible de passer cet argument soit comme un `float` (ex. `0.1`), soit comme un `tuple(float, float)` pour spécifier un intervalle d'écart-type à échantillonner aléatoirement pour chaque batch (ex. `(0.05, 0.2)`). (default: `0.1`)

```bash
  --perturbation-std PERTURBATION_STD
```

* L'écart-type pour l'augmentation par mise à l'échelle de la boîte de simulation. Si cet argument est défini, une augmentation par mise à l'échelle de la boîte de simulation sera appliquée aux données d'entraînement avec l'écart-type spécifié. Sinon, aucune augmentation par mise à l'échelle de la boîte de simulation ne sera appliquée. Il est possible de passer cet argument soit comme un `float` (ex. `0.1`), soit comme un `tuple(float, float)` pour spécifier un intervalle d'écart-type à échantillonner aléatoirement pour chaque batch (ex. `(0.05, 0.2)`). (default: `0.1`)

```bash
  --box-scaling-std BOX_SCALING_STD
```

* L'écart-type pour l'augmentation par cisaillement de la boîte de simulation. Si cet argument est défini, une augmentation par cisaillement de la boîte de simulation sera appliquée aux données d'entraînement avec l'écart-type spécifié. Sinon, aucune augmentation par cisaillement de la boîte de simulation ne sera appliquée. Il est possible de passer cet argument soit comme un `float` (ex. `0.1`), soit comme un `tuple(float, float)` pour spécifier un intervalle d'écart-type à échantillonner aléatoirement pour chaque batch (ex. `(0.05, 0.2)`). (default: `0.1`)

```bash
  --box-shearing-std BOX_SHEARING_STD
```

## Callbacks et monitoring

* activer l'early stopping. (default: `False`)

```bash
  --early-stopping
```

* patience pour l'early stopping (en epochs). (default: `20`)

```bash
  --early-stopping-patience EARLY_STOPPING_PATIENCE
```

* nombre de modèles à sauvegarder selon la performance sur le set de validation. (default: `3`)

```bash
  --save-top-k SAVE_TOP_K
```

* où enregistrer les checkpoints. (default: `None` pour utiliser les logs par défaut de Lightning)
  
```bash
  --checkpoint-dir CHECKPOINT_DIR
```

## Logging

* Où enregistrer les logs et les sorties de TensorBoard. (default: `logs`)

```bash
  --log-dir LOG_DIR
```

* nom à utiliser pour stocker les résultats du script d'entraînement. Je crois qu'avec TensorBoard, la terminologie adaptée est "experiment" ? (default: `None` pour auto-générer un nom d'expérience basé sur la date et l'heure)  

```bash
  --experiment-name EXPERIMENT_NAME
```

* Fréquence de logging (en steps). (default: `50`)

```bash
  --log-every-n-steps LOG_EVERY_N_STEPS
```

## Reproductibilité

* seed pour le générateur de nombres aléatoires (default: `42`)

```bash
  --seed SEED
```

* activer le mode déterministe ([peut réduire les performances?](https://discuss.pytorch.org/t/performance-regression-in-torch-2-0-with-deterministic-algorithms/188690/2)). (default: `False`)

```bash
  --deterministic
```

## HPO

* réutiliser les résultats d'une HPO précédente pour entraîner un modèle avec les meilleurs hyperparamètres trouvés. (default: `False`, fournir un lien type `sqlite:///optuna.db`)

```bash
  --storage STORAGE
```

## Gestion des checkpoints

* reprendre l'entraînement à partir d'un checkpoint existant. (default: `None`)

```bash
  --resume-from RESUME_FROM
```

* autoriser le chargement de checkpoints non sécurisés (`torch.load(weights_only=False)`) -- à utiliser avec précaution et uniquement pour les checkpoints de confiance, car cela peut exposer à des risques de sécurité. (default: `False`)

```bash
  --allow-unsafe-checkpoint-loading
```

* enregistrer les détails de la configuration d'entraînement dans un fichier JSON pour faciliter la reproductibilité et le suivi des expériences. (default: `True`)

```bash
  --save-config         Save training configuration to JSON file. (default: True)
```

* charger une configuration à partir d'un fichier JSON ou YAML. (default: `None`)

```bash
  --config CONFIG       Load configuration from JSON/YAML file. (default: None)
```

## Performance

* Précision de l'entraînement. Auto-détectée si non spécifiée. (default: `None`)

```bash
  --precision {32,16,bf16,16-mixed,bf16-mixed}
```

* matériel à utiliser (ex. '1' pour GPU 1, 'auto' pour tous les devices disponibles). (default: `auto`)

```bash
  --devices DEVICES
```

> **Note** : on n'a pas testé l'entraînement multi-GPU (je crois ?), mais la plupart des codes fournissent une option de ce genre.

* utiliser CPU/GPU ? (ou même TPU si disponible ?). (default: `auto` pour détecter automatiquement le matériel disponible)

```bash
  --accelerator {auto,cpu,gpu,tpu}
                        Accelerator type. (default: auto)
```
