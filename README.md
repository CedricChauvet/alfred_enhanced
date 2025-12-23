Pour commencer, nous telechargeons le modele proposé par le repo:

# https://github.com/askforalfred/alfred

## charger les chemins:
commencer par conda activate alfred_env,
puis:   
/media/cedrix/Ubuntu_2To/Alfred/alfred_experiments$ source .env

"""
✓ ALFRED environment loaded
  ALFRED_ROOT: /media/cedrix/Ubuntu_2To/Alfred/alfred
  ALFRED_EXP_ROOT: /media/cedrix/Ubuntu_2To/Alfred/alfred_experiments
  PYTHONPATH: /media/cedrix/Ubuntu_2To/Alfred/alfred:/media/cedrix/Ubuntu_2To/Alfred/alfred_experiments:
"""

Durant nos experiences, nous travaillerons exclusivement sur ALFRED_EXP_ROOT
ALFRED_ROOT doit etre la copie exacte  du repo github

## importer le dataset

    j'ai 2 dataset installables via :
    sh download_data.sh json_feat
     ou 
    sh download_data.sh full

    on peut travailler avec json_feat tant qu'on n'entraine pas la partie visuelle (full contient notamment les image RGB en plus)
    MAIS, j'ai des soucis avec full.

##  Preprocess:

    Lors du premier lancement de d'un script  il faut preprocesser le dataset:

    --preprocess 
    cela completera le dataset avec un dossier pp pour chaque traj avec des fichiers ann_*.json


## 1 modele de baseline
 wget https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/seq2seq_pm_chkpt.zip

afin de verifier que les scripts d'evaluation fonctionne et que tout est bien installé, c'est un peu une demo a ce stade.

python models/eval/eval_seq2seq.py \
  --model_path /media/cedrix/Ubuntu_2To/Alfred/alfred_experiments/experiments/Baseline/best_seen.pth \
  --eval_split valid_seen \
  --data data/json_feat_2.1.0 \
  --model models.model.seq2seq_im_mask \
  --gpu \
  --num_threads 2
  --preprocess  (une fois)

puis on va réentrainer le baseline avec la commande train.sh avec un yaml de config



### resultats du modele baseline:

pour le modele téléchrgé
-------------
SR: 8/820 = 0.010
GC: 140/2109 = 0.066
PLW SR: 0.003
PLW GC: 0.038
-------------

## training du baseline:
On utilise le script train.sh et on créé le yaml baseline_reproduction

cd $ALFRED_EXP_ROOT
./scripts/train.sh ./config/baseline_reproduction.yaml

Les resultats du training sont stockés dans $ALFRED_EXP_ROOT/experiments/

-------------
SR: 19/820 = 0.023
GC: 194/2109 = 0.092
PLW SR: 0.018
PLW GC: 0.073
-------------

# Amélioration du modele

    ## Chain oh Thoughts (CoT)

    lien vers le readme de CoT (models/model/CoT)