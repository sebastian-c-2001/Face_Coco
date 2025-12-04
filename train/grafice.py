import re
import matplotlib.pyplot as plt

# Am copiat întregul tău log aici

log_data_pre = """
C:\PycharmProjects\.venv\Scripts\python.exe C:\PycharmProjects\Face_Coco\train\train.py 
Se folosește dispozitivul: cuda
Se încarcă modelul ResNet-18 pre-antrenat...
--- Începe Antrenamentul ---
Epoca 1/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:08<00:00,  5.69it/s]
Epoca 1/100 [Val]: 100%|██████████| 129/129 [00:34<00:00,  3.75it/s]

Epoca 1/100 | Train Loss: 0.115384 | Train MAE: 0.152832 | Val Loss: 0.017534 | Val MAE: 0.098024
  Val Loss s-a îmbunătățit (inf --> 0.017534). Se salvează modelul...
Epoca 2/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:08<00:00,  5.67it/s]
Epoca 2/100 [Val]: 100%|██████████| 129/129 [00:41<00:00,  3.11it/s]

Epoca 2/100 | Train Loss: 0.016908 | Train MAE: 0.095833 | Val Loss: 0.019205 | Val MAE: 0.098482
Epoca 3/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:25<00:00,  4.56it/s]
Epoca 3/100 [Val]: 100%|██████████| 129/129 [00:36<00:00,  3.51it/s]

Epoca 3/100 | Train Loss: 0.014317 | Train MAE: 0.087266 | Val Loss: 0.014093 | Val MAE: 0.082973
  Val Loss s-a îmbunătățit (0.017534 --> 0.014093). Se salvează modelul...
Epoca 4/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:08<00:00,  5.68it/s]
Epoca 4/100 [Val]: 100%|██████████| 129/129 [00:34<00:00,  3.78it/s]

Epoca 4/100 | Train Loss: 0.009300 | Train MAE: 0.070251 | Val Loss: 0.012095 | Val MAE: 0.073530
  Val Loss s-a îmbunătățit (0.014093 --> 0.012095). Se salvează modelul...
Epoca 5/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:08<00:00,  5.66it/s]
Epoca 5/100 [Val]: 100%|██████████| 129/129 [00:34<00:00,  3.79it/s]

Epoca 5/100 | Train Loss: 0.006373 | Train MAE: 0.056949 | Val Loss: 0.005180 | Val MAE: 0.052126
  Val Loss s-a îmbunătățit (0.012095 --> 0.005180). Se salvează modelul...
Epoca 6/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:08<00:00,  5.68it/s]
Epoca 6/100 [Val]: 100%|██████████| 129/129 [00:34<00:00,  3.77it/s]

Epoca 6/100 | Train Loss: 0.005077 | Train MAE: 0.050075 | Val Loss: 0.006824 | Val MAE: 0.050164
Epoca 7/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.76it/s]
Epoca 7/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.85it/s]
Epoca 8/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 7/100 | Train Loss: 0.007461 | Train MAE: 0.058349 | Val Loss: 0.011578 | Val MAE: 0.076479
Epoca 8/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.80it/s]
Epoca 8/100 [Val]: 100%|██████████| 129/129 [00:35<00:00,  3.68it/s]
Epoca 9/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 8/100 | Train Loss: 0.006127 | Train MAE: 0.054619 | Val Loss: 0.006604 | Val MAE: 0.058087
Epoca 9/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.75it/s]
Epoca 9/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.80it/s]

Epoca 9/100 | Train Loss: 0.004212 | Train MAE: 0.044758 | Val Loss: 0.003517 | Val MAE: 0.039047
  Val Loss s-a îmbunătățit (0.005180 --> 0.003517). Se salvează modelul...
Epoca 10/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]

Epoca 10/100 | Train Loss: 0.003676 | Train MAE: 0.041797 | Val Loss: 0.002933 | Val MAE: 0.035687
  Val Loss s-a îmbunătățit (0.003517 --> 0.002933). Se salvează modelul...
Epoca 11/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:11<00:00,  5.42it/s]
Epoca 11/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.82it/s]
Epoca 12/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 11/100 | Train Loss: 0.003054 | Train MAE: 0.037283 | Val Loss: 0.003050 | Val MAE: 0.039232
Epoca 12/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:09<00:00,  5.62it/s]
Epoca 12/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.41it/s]

Epoca 12/100 | Train Loss: 0.002788 | Train MAE: 0.035288 | Val Loss: 0.002540 | Val MAE: 0.034165
  Val Loss s-a îmbunătățit (0.002933 --> 0.002540). Se salvează modelul...
Epoca 13/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.11it/s]
Epoca 13/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.36it/s]
Epoca 14/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 13/100 | Train Loss: 0.002573 | Train MAE: 0.034229 | Val Loss: 0.002629 | Val MAE: 0.035136
Epoca 14/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:02<00:00,  6.20it/s]
Epoca 14/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.35it/s]

Epoca 14/100 | Train Loss: 0.002220 | Train MAE: 0.031681 | Val Loss: 0.002390 | Val MAE: 0.033446
  Val Loss s-a îmbunătățit (0.002540 --> 0.002390). Se salvează modelul...
Epoca 15/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.16it/s]
Epoca 15/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.10it/s]

Epoca 15/100 | Train Loss: 0.001946 | Train MAE: 0.030010 | Val Loss: 0.002229 | Val MAE: 0.032715
  Val Loss s-a îmbunătățit (0.002390 --> 0.002229). Se salvează modelul...
Epoca 16/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.89it/s]
Epoca 16/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.13it/s]

Epoca 16/100 | Train Loss: 0.001740 | Train MAE: 0.028675 | Val Loss: 0.002048 | Val MAE: 0.029193
  Val Loss s-a îmbunătățit (0.002229 --> 0.002048). Se salvează modelul...
Epoca 17/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.08it/s]
Epoca 17/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.26it/s]

Epoca 17/100 | Train Loss: 0.001578 | Train MAE: 0.027369 | Val Loss: 0.001717 | Val MAE: 0.026875
  Val Loss s-a îmbunătățit (0.002048 --> 0.001717). Se salvează modelul...
Epoca 18/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.02it/s]
Epoca 18/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.16it/s]

Epoca 18/100 | Train Loss: 0.001597 | Train MAE: 0.027615 | Val Loss: 0.002962 | Val MAE: 0.038208
Epoca 19/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.78it/s]
Epoca 19/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  3.93it/s]

Epoca 19/100 | Train Loss: 0.001505 | Train MAE: 0.026969 | Val Loss: 0.001547 | Val MAE: 0.025740
  Val Loss s-a îmbunătățit (0.001717 --> 0.001547). Se salvează modelul...
Epoca 20/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.85it/s]
Epoca 20/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  3.94it/s]
Epoca 21/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 20/100 | Train Loss: 0.001245 | Train MAE: 0.024760 | Val Loss: 0.001639 | Val MAE: 0.027419
Epoca 21/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.77it/s]
Epoca 21/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.56it/s]

Epoca 21/100 | Train Loss: 0.001623 | Train MAE: 0.027733 | Val Loss: 0.001635 | Val MAE: 0.027465
Epoca 22/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:02<00:00,  6.23it/s]
Epoca 22/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.19it/s]
Epoca 23/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 22/100 | Train Loss: 0.001173 | Train MAE: 0.024228 | Val Loss: 0.001638 | Val MAE: 0.025910
Epoca 23/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.73it/s]
Epoca 23/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.86it/s]

Epoca 23/100 | Train Loss: 0.001038 | Train MAE: 0.022873 | Val Loss: 0.001462 | Val MAE: 0.025632
  Val Loss s-a îmbunătățit (0.001547 --> 0.001462). Se salvează modelul...
Epoca 24/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.76it/s]
Epoca 24/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.80it/s]

Epoca 24/100 | Train Loss: 0.000951 | Train MAE: 0.022199 | Val Loss: 0.001238 | Val MAE: 0.022252
  Val Loss s-a îmbunătățit (0.001462 --> 0.001238). Se salvează modelul...
Epoca 25/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:08<00:00,  5.70it/s]
Epoca 25/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.84it/s]

Epoca 25/100 | Train Loss: 0.001267 | Train MAE: 0.025208 | Val Loss: 0.001656 | Val MAE: 0.027271
Epoca 26/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.85it/s]
Epoca 26/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.85it/s]
Epoca 27/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 26/100 | Train Loss: 0.001022 | Train MAE: 0.022821 | Val Loss: 0.001474 | Val MAE: 0.025433
Epoca 27/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.82it/s]
Epoca 27/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.91it/s]

Epoca 27/100 | Train Loss: 0.000900 | Train MAE: 0.021642 | Val Loss: 0.001423 | Val MAE: 0.024299
Epoca 28/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.77it/s]
Epoca 28/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.86it/s]

Epoca 28/100 | Train Loss: 0.000693 | Train MAE: 0.019213 | Val Loss: 0.001377 | Val MAE: 0.025309
Epoca 29/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.75it/s]
Epoca 29/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.84it/s]
Epoca 30/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 29/100 | Train Loss: 0.000676 | Train MAE: 0.019137 | Val Loss: 0.001244 | Val MAE: 0.022533
Epoca 30/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.82it/s]
Epoca 30/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.85it/s]
Epoca 31/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 30/100 | Train Loss: 0.001076 | Train MAE: 0.022868 | Val Loss: 0.002595 | Val MAE: 0.031933
Epoca 31/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.73it/s]
Epoca 31/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.85it/s]
Epoca 32/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 31/100 | Train Loss: 0.001519 | Train MAE: 0.026312 | Val Loss: 0.002050 | Val MAE: 0.029687
Epoca 32/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.83it/s]
Epoca 32/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.39it/s]

Epoca 32/100 | Train Loss: 0.001136 | Train MAE: 0.023558 | Val Loss: 0.001352 | Val MAE: 0.024620
Epoca 33/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:02<00:00,  6.19it/s]
Epoca 33/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.25it/s]

Epoca 33/100 | Train Loss: 0.000792 | Train MAE: 0.020480 | Val Loss: 0.001213 | Val MAE: 0.022475
  Val Loss s-a îmbunătățit (0.001238 --> 0.001213). Se salvează modelul...
Epoca 34/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.14it/s]
Epoca 34/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.33it/s]

Epoca 34/100 | Train Loss: 0.000648 | Train MAE: 0.018797 | Val Loss: 0.001198 | Val MAE: 0.022375
  Val Loss s-a îmbunătățit (0.001213 --> 0.001198). Se salvează modelul...
Epoca 35/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:17<00:00,  5.02it/s]
Epoca 35/100 [Val]: 100%|██████████| 129/129 [00:46<00:00,  2.80it/s]

Epoca 35/100 | Train Loss: 0.000588 | Train MAE: 0.017901 | Val Loss: 0.001166 | Val MAE: 0.021958
  Val Loss s-a îmbunătățit (0.001198 --> 0.001166). Se salvează modelul...
Epoca 36/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:15<00:00,  5.16it/s]
Epoca 36/100 [Val]: 100%|██████████| 129/129 [00:37<00:00,  3.40it/s]
Epoca 37/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 36/100 | Train Loss: 0.000597 | Train MAE: 0.018039 | Val Loss: 0.001257 | Val MAE: 0.022943
Epoca 37/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:14<00:00,  5.19it/s]
Epoca 37/100 [Val]: 100%|██████████| 129/129 [00:47<00:00,  2.69it/s]
Epoca 38/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 37/100 | Train Loss: 0.000548 | Train MAE: 0.017310 | Val Loss: 0.001252 | Val MAE: 0.023064
Epoca 38/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:31<00:00,  4.25it/s]
Epoca 38/100 [Val]: 100%|██████████| 129/129 [00:53<00:00,  2.39it/s]
Epoca 39/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 38/100 | Train Loss: 0.000522 | Train MAE: 0.017023 | Val Loss: 0.001306 | Val MAE: 0.023977
Epoca 39/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:18<00:00,  4.94it/s]
Epoca 39/100 [Val]: 100%|██████████| 129/129 [00:42<00:00,  3.05it/s]

Epoca 39/100 | Train Loss: 0.000520 | Train MAE: 0.016951 | Val Loss: 0.001141 | Val MAE: 0.021992
  Val Loss s-a îmbunătățit (0.001166 --> 0.001141). Se salvează modelul...
Epoca 40/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:20<00:00,  4.82it/s]
Epoca 40/100 [Val]: 100%|██████████| 129/129 [00:39<00:00,  3.29it/s]
Epoca 41/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 40/100 | Train Loss: 0.000547 | Train MAE: 0.017345 | Val Loss: 0.001332 | Val MAE: 0.024420
Epoca 41/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:12<00:00,  5.38it/s]
Epoca 41/100 [Val]: 100%|██████████| 129/129 [00:40<00:00,  3.19it/s]
Epoca 42/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 41/100 | Train Loss: 0.000700 | Train MAE: 0.019260 | Val Loss: 0.001173 | Val MAE: 0.021937
Epoca 42/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:08<00:00,  5.70it/s]
Epoca 42/100 [Val]: 100%|██████████| 129/129 [00:34<00:00,  3.77it/s]

Epoca 42/100 | Train Loss: 0.000450 | Train MAE: 0.015889 | Val Loss: 0.001204 | Val MAE: 0.023102
Epoca 43/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:08<00:00,  5.70it/s]
Epoca 43/100 [Val]: 100%|██████████| 129/129 [00:36<00:00,  3.52it/s]
Epoca 44/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 43/100 | Train Loss: 0.000398 | Train MAE: 0.015060 | Val Loss: 0.001171 | Val MAE: 0.022319
Epoca 44/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.81it/s]
Epoca 44/100 [Val]: 100%|██████████| 129/129 [00:34<00:00,  3.73it/s]
Epoca 45/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 44/100 | Train Loss: 0.000404 | Train MAE: 0.015159 | Val Loss: 0.001176 | Val MAE: 0.022396
Epoca 45/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.81it/s]
Epoca 45/100 [Val]: 100%|██████████| 129/129 [00:37<00:00,  3.44it/s]

Epoca 45/100 | Train Loss: 0.000395 | Train MAE: 0.014843 | Val Loss: 0.001109 | Val MAE: 0.021408
  Val Loss s-a îmbunătățit (0.001141 --> 0.001109). Se salvează modelul...
Epoca 46/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:12<00:00,  5.35it/s]
Epoca 46/100 [Val]: 100%|██████████| 129/129 [00:37<00:00,  3.40it/s]
Epoca 47/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 46/100 | Train Loss: 0.000347 | Train MAE: 0.014069 | Val Loss: 0.001143 | Val MAE: 0.022102
Epoca 47/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:09<00:00,  5.58it/s]
Epoca 47/100 [Val]: 100%|██████████| 129/129 [00:41<00:00,  3.15it/s]
Epoca 48/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 47/100 | Train Loss: 0.000379 | Train MAE: 0.014484 | Val Loss: 0.001119 | Val MAE: 0.021181
Epoca 48/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:12<00:00,  5.40it/s]
Epoca 48/100 [Val]: 100%|██████████| 129/129 [00:40<00:00,  3.20it/s]
Epoca 49/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 48/100 | Train Loss: 0.000344 | Train MAE: 0.013922 | Val Loss: 0.001154 | Val MAE: 0.021966
Epoca 49/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:11<00:00,  5.46it/s]
Epoca 49/100 [Val]: 100%|██████████| 129/129 [00:38<00:00,  3.39it/s]

Epoca 49/100 | Train Loss: 0.000330 | Train MAE: 0.013707 | Val Loss: 0.001189 | Val MAE: 0.022043
Epoca 50/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:10<00:00,  5.52it/s]
Epoca 50/100 [Val]: 100%|██████████| 129/129 [00:37<00:00,  3.45it/s]

Epoca 50/100 | Train Loss: 0.000330 | Train MAE: 0.013633 | Val Loss: 0.001097 | Val MAE: 0.021414
  Val Loss s-a îmbunătățit (0.001109 --> 0.001097). Se salvează modelul...
Epoca 51/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:12<00:00,  5.34it/s]
Epoca 51/100 [Val]: 100%|██████████| 129/129 [00:37<00:00,  3.47it/s]

Epoca 51/100 | Train Loss: 0.000313 | Train MAE: 0.013339 | Val Loss: 0.001088 | Val MAE: 0.021067
  Val Loss s-a îmbunătățit (0.001097 --> 0.001088). Se salvează modelul...
Epoca 52/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:10<00:00,  5.53it/s]
Epoca 52/100 [Val]: 100%|██████████| 129/129 [00:37<00:00,  3.40it/s]

Epoca 52/100 | Train Loss: 0.000293 | Train MAE: 0.012958 | Val Loss: 0.001055 | Val MAE: 0.020888
  Val Loss s-a îmbunătățit (0.001088 --> 0.001055). Se salvează modelul...
Epoca 53/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:10<00:00,  5.51it/s]
Epoca 53/100 [Val]: 100%|██████████| 129/129 [00:37<00:00,  3.46it/s]

Epoca 53/100 | Train Loss: 0.000290 | Train MAE: 0.012838 | Val Loss: 0.001090 | Val MAE: 0.021345
Epoca 54/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:10<00:00,  5.48it/s]
Epoca 54/100 [Val]: 100%|██████████| 129/129 [00:37<00:00,  3.43it/s]
Epoca 55/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 54/100 | Train Loss: 0.000272 | Train MAE: 0.012474 | Val Loss: 0.001215 | Val MAE: 0.023110
Epoca 55/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:09<00:00,  5.63it/s]
Epoca 55/100 [Val]: 100%|██████████| 129/129 [00:37<00:00,  3.40it/s]
Epoca 56/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 55/100 | Train Loss: 0.000320 | Train MAE: 0.013351 | Val Loss: 0.001067 | Val MAE: 0.020729
Epoca 56/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.87it/s]
Epoca 56/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.07it/s]
Epoca 57/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 56/100 | Train Loss: 0.000258 | Train MAE: 0.012153 | Val Loss: 0.001124 | Val MAE: 0.021743
Epoca 57/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.77it/s]
Epoca 57/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  4.01it/s]
Epoca 58/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 57/100 | Train Loss: 0.000219 | Train MAE: 0.011281 | Val Loss: 0.001071 | Val MAE: 0.020530
Epoca 58/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.94it/s]
Epoca 58/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.08it/s]

Epoca 58/100 | Train Loss: 0.000228 | Train MAE: 0.011458 | Val Loss: 0.001054 | Val MAE: 0.020390
  Val Loss s-a îmbunătățit (0.001055 --> 0.001054). Se salvează modelul...
Epoca 59/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.95it/s]
Epoca 59/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  4.01it/s]
Epoca 60/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 59/100 | Train Loss: 0.000219 | Train MAE: 0.011258 | Val Loss: 0.001084 | Val MAE: 0.021203
Epoca 60/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 60/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.86it/s]

Epoca 60/100 | Train Loss: 0.000209 | Train MAE: 0.011009 | Val Loss: 0.001024 | Val MAE: 0.020625
  Val Loss s-a îmbunătățit (0.001054 --> 0.001024). Se salvează modelul...
Epoca 61/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.04it/s]
Epoca 61/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.15it/s]

Epoca 61/100 | Train Loss: 0.000208 | Train MAE: 0.010959 | Val Loss: 0.001079 | Val MAE: 0.020950
Epoca 62/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.06it/s]
Epoca 62/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.88it/s]

Epoca 62/100 | Train Loss: 0.000208 | Train MAE: 0.010946 | Val Loss: 0.001050 | Val MAE: 0.020756
Epoca 63/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.03it/s]
Epoca 63/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  4.03it/s]

Epoca 63/100 | Train Loss: 0.000196 | Train MAE: 0.010596 | Val Loss: 0.001014 | Val MAE: 0.020267
  Val Loss s-a îmbunătățit (0.001024 --> 0.001014). Se salvează modelul...
Epoca 64/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.97it/s]
Epoca 64/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  3.92it/s]
Epoca 65/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 64/100 | Train Loss: 0.000172 | Train MAE: 0.010019 | Val Loss: 0.001070 | Val MAE: 0.021171
Epoca 65/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.01it/s]
Epoca 65/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.13it/s]

Epoca 65/100 | Train Loss: 0.000197 | Train MAE: 0.010666 | Val Loss: 0.001065 | Val MAE: 0.020931
Epoca 66/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.03it/s]
Epoca 66/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.13it/s]

Epoca 66/100 | Train Loss: 0.000183 | Train MAE: 0.010234 | Val Loss: 0.001155 | Val MAE: 0.022378
Epoca 67/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.81it/s]
Epoca 67/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  3.93it/s]

Epoca 67/100 | Train Loss: 0.000200 | Train MAE: 0.010735 | Val Loss: 0.001004 | Val MAE: 0.020293
  Val Loss s-a îmbunătățit (0.001014 --> 0.001004). Se salvează modelul...
Epoca 68/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.95it/s]
Epoca 68/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.91it/s]

Epoca 68/100 | Train Loss: 0.000191 | Train MAE: 0.010441 | Val Loss: 0.000986 | Val MAE: 0.020061
  Val Loss s-a îmbunătățit (0.001004 --> 0.000986). Se salvează modelul...
Epoca 69/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.95it/s]
Epoca 69/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.13it/s]
Epoca 70/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 69/100 | Train Loss: 0.000174 | Train MAE: 0.009982 | Val Loss: 0.001048 | Val MAE: 0.021236
Epoca 70/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.89it/s]
Epoca 70/100 [Val]: 100%|██████████| 129/129 [00:41<00:00,  3.08it/s]
Epoca 71/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 70/100 | Train Loss: 0.000171 | Train MAE: 0.009917 | Val Loss: 0.001001 | Val MAE: 0.020197
Epoca 71/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.96it/s]
Epoca 71/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  3.97it/s]
Epoca 72/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 71/100 | Train Loss: 0.000162 | Train MAE: 0.009617 | Val Loss: 0.001032 | Val MAE: 0.020507
Epoca 72/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.97it/s]
Epoca 72/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  3.94it/s]
Epoca 73/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 72/100 | Train Loss: 0.000163 | Train MAE: 0.009687 | Val Loss: 0.001029 | Val MAE: 0.020702
Epoca 73/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:25<00:00,  4.55it/s]
Epoca 73/100 [Val]: 100%|██████████| 129/129 [00:50<00:00,  2.57it/s]

Epoca 73/100 | Train Loss: 0.000204 | Train MAE: 0.010592 | Val Loss: 0.001039 | Val MAE: 0.020561
Epoca 74/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:29<00:00,  4.34it/s]
Epoca 74/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  4.00it/s]
Epoca 75/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 74/100 | Train Loss: 0.000150 | Train MAE: 0.009286 | Val Loss: 0.001057 | Val MAE: 0.021043
Epoca 75/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 75/100 [Val]: 100%|██████████| 129/129 [00:34<00:00,  3.69it/s]

Epoca 75/100 | Train Loss: 0.000134 | Train MAE: 0.008808 | Val Loss: 0.000991 | Val MAE: 0.019983
Epoca 76/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.96it/s]
Epoca 76/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  3.94it/s]
Epoca 77/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 76/100 | Train Loss: 0.000134 | Train MAE: 0.008763 | Val Loss: 0.001013 | Val MAE: 0.019989
Epoca 77/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:02<00:00,  6.20it/s]
Epoca 77/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.34it/s]
Epoca 78/100 [Train] LR=1.0e-02:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 77/100 | Train Loss: 0.000128 | Train MAE: 0.008648 | Val Loss: 0.000999 | Val MAE: 0.020052
Epoca 78/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.15it/s]
Epoca 78/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.30it/s]
Epoca 79/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 78/100 | Train Loss: 0.000125 | Train MAE: 0.008526 | Val Loss: 0.001009 | Val MAE: 0.019963
  Răbdarea a expirat (după 10 epoci). Se scade Learning Rate-ul.
  Noul Learning Rate: 0.001
Epoca 79/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:04<00:00,  6.03it/s]
Epoca 79/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.16it/s]

Epoca 79/100 | Train Loss: 0.000073 | Train MAE: 0.006420 | Val Loss: 0.000958 | Val MAE: 0.019384
  Val Loss s-a îmbunătățit (0.000986 --> 0.000958). Se salvează modelul...
Epoca 80/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:05<00:00,  5.96it/s]
Epoca 80/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.85it/s]

Epoca 80/100 | Train Loss: 0.000057 | Train MAE: 0.005739 | Val Loss: 0.000958 | Val MAE: 0.019367
  Val Loss s-a îmbunătățit (0.000958 --> 0.000958). Se salvează modelul...
Epoca 81/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 81/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.30it/s]

Epoca 81/100 | Train Loss: 0.000050 | Train MAE: 0.005453 | Val Loss: 0.000945 | Val MAE: 0.019295
  Val Loss s-a îmbunătățit (0.000958 --> 0.000945). Se salvează modelul...
Epoca 82/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:04<00:00,  5.98it/s]
Epoca 82/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.05it/s]

Epoca 82/100 | Train Loss: 0.000046 | Train MAE: 0.005247 | Val Loss: 0.000948 | Val MAE: 0.019297
Epoca 83/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:07<00:00,  5.80it/s]
Epoca 83/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.27it/s]

Epoca 83/100 | Train Loss: 0.000043 | Train MAE: 0.005093 | Val Loss: 0.000948 | Val MAE: 0.019318
Epoca 84/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:04<00:00,  6.07it/s]
Epoca 84/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.12it/s]
Epoca 85/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 84/100 | Train Loss: 0.000041 | Train MAE: 0.004972 | Val Loss: 0.000950 | Val MAE: 0.019338
Epoca 85/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 85/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.24it/s]

Epoca 85/100 | Train Loss: 0.000039 | Train MAE: 0.004868 | Val Loss: 0.000951 | Val MAE: 0.019312
Epoca 86/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.16it/s]
Epoca 86/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.17it/s]

Epoca 86/100 | Train Loss: 0.000037 | Train MAE: 0.004751 | Val Loss: 0.000954 | Val MAE: 0.019382
Epoca 87/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.11it/s]
Epoca 87/100 [Val]: 100%|██████████| 129/129 [00:41<00:00,  3.14it/s]
Epoca 88/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 87/100 | Train Loss: 0.000036 | Train MAE: 0.004663 | Val Loss: 0.000949 | Val MAE: 0.019341
Epoca 88/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:12<00:00,  5.40it/s]
Epoca 88/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.23it/s]

Epoca 88/100 | Train Loss: 0.000035 | Train MAE: 0.004583 | Val Loss: 0.000948 | Val MAE: 0.019338
Epoca 89/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.13it/s]
Epoca 89/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.28it/s]

Epoca 89/100 | Train Loss: 0.000034 | Train MAE: 0.004532 | Val Loss: 0.000945 | Val MAE: 0.019310
  Val Loss s-a îmbunătățit (0.000945 --> 0.000945). Se salvează modelul...
Epoca 90/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.11it/s]
Epoca 90/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.31it/s]
Epoca 91/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 90/100 | Train Loss: 0.000032 | Train MAE: 0.004421 | Val Loss: 0.000948 | Val MAE: 0.019333
Epoca 91/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.20it/s]
Epoca 91/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.31it/s]
Epoca 92/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 91/100 | Train Loss: 0.000032 | Train MAE: 0.004407 | Val Loss: 0.000949 | Val MAE: 0.019343
Epoca 92/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.23it/s]
Epoca 92/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.37it/s]

Epoca 92/100 | Train Loss: 0.000032 | Train MAE: 0.004387 | Val Loss: 0.000947 | Val MAE: 0.019329
Epoca 93/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 93/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.29it/s]
Epoca 94/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 93/100 | Train Loss: 0.000031 | Train MAE: 0.004359 | Val Loss: 0.000951 | Val MAE: 0.019370
Epoca 94/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.19it/s]
Epoca 94/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.33it/s]
Epoca 95/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 94/100 | Train Loss: 0.000030 | Train MAE: 0.004288 | Val Loss: 0.000951 | Val MAE: 0.019371
Epoca 95/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.16it/s]
Epoca 95/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.33it/s]

Epoca 95/100 | Train Loss: 0.000029 | Train MAE: 0.004200 | Val Loss: 0.000954 | Val MAE: 0.019378
Epoca 96/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.18it/s]
Epoca 96/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.33it/s]
Epoca 97/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 96/100 | Train Loss: 0.000029 | Train MAE: 0.004214 | Val Loss: 0.000955 | Val MAE: 0.019389
Epoca 97/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.19it/s]
Epoca 97/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.36it/s]
Epoca 98/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 97/100 | Train Loss: 0.000028 | Train MAE: 0.004150 | Val Loss: 0.000948 | Val MAE: 0.019354
Epoca 98/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.20it/s]
Epoca 98/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.33it/s]
Epoca 99/100 [Train] LR=1.0e-03:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 98/100 | Train Loss: 0.000027 | Train MAE: 0.004075 | Val Loss: 0.000951 | Val MAE: 0.019400
Epoca 99/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.17it/s]
Epoca 99/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.37it/s]
Epoca 100/100 [Train] LR=1.0e-04:    0%|          | 0/389 [00:00<?, ?it/s]
Epoca 99/100 | Train Loss: 0.000028 | Train MAE: 0.004095 | Val Loss: 0.000951 | Val MAE: 0.019420
  Răbdarea a expirat (după 10 epoci). Se scade Learning Rate-ul.
  Noul Learning Rate: 0.0001
Epoca 100/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.18it/s]
Epoca 100/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.38it/s]

Epoca 100/100 | Train Loss: 0.000024 | Train MAE: 0.003849 | Val Loss: 0.000948 | Val MAE: 0.019344

--- Antrenament Finalizat ---
Cel mai bun model a fost salvat în: best_model_face_keypoints.pth (Val Loss: 0.000945)

Process finished with exit code 0
"""
log_data= """
C:\PycharmProjects\.venv\Scripts\python.exe C:\PycharmProjects\Face_Coco\train\train.py 
Se folosește dispozitivul: cuda
Începe antrenarea de la 0
--- Începe Antrenamentul ---
Epoca 1/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:19<00:00,  4.88it/s]
Epoca 1/100 [Val]: 100%|██████████| 129/129 [00:39<00:00,  3.24it/s]

Epoca 1/100 | Train Loss: 0.087333 | Train MAE: 0.131776 | Val Loss: 0.019638 | Val MAE: 0.100349
  Val Loss s-a îmbunătățit (inf --> 0.019638). Se salvează modelul...
Epoca 2/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.14it/s]
Epoca 2/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.08it/s]

Epoca 2/100 | Train Loss: 0.017021 | Train MAE: 0.095852 | Val Loss: 0.017102 | Val MAE: 0.093477
  Val Loss s-a îmbunătățit (0.019638 --> 0.017102). Se salvează modelul...
Epoca 3/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:02<00:00,  6.18it/s]
Epoca 3/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.16it/s]

Epoca 3/100 | Train Loss: 0.012085 | Train MAE: 0.079988 | Val Loss: 0.009890 | Val MAE: 0.069443
  Val Loss s-a îmbunătățit (0.017102 --> 0.009890). Se salvează modelul...
Epoca 4/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.79it/s]
Epoca 4/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.10it/s]

Epoca 4/100 | Train Loss: 0.009013 | Train MAE: 0.068706 | Val Loss: 0.008813 | Val MAE: 0.065525
  Val Loss s-a îmbunătățit (0.009890 --> 0.008813). Se salvează modelul...
Epoca 5/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.07it/s]
Epoca 5/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.24it/s]

Epoca 5/100 | Train Loss: 0.006864 | Train MAE: 0.059088 | Val Loss: 0.007101 | Val MAE: 0.058310
  Val Loss s-a îmbunătățit (0.008813 --> 0.007101). Se salvează modelul...
Epoca 6/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.08it/s]
Epoca 6/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.26it/s]

Epoca 6/100 | Train Loss: 0.005350 | Train MAE: 0.051401 | Val Loss: 0.007116 | Val MAE: 0.054877
Epoca 7/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 7/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.19it/s]

Epoca 7/100 | Train Loss: 0.004586 | Train MAE: 0.046748 | Val Loss: 0.017367 | Val MAE: 0.061058
Epoca 8/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.08it/s]
Epoca 8/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.19it/s]

Epoca 8/100 | Train Loss: 0.004054 | Train MAE: 0.043629 | Val Loss: 0.003650 | Val MAE: 0.038928
  Val Loss s-a îmbunătățit (0.007101 --> 0.003650). Se salvează modelul...
Epoca 9/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.98it/s]
Epoca 9/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.25it/s]

Epoca 9/100 | Train Loss: 0.003428 | Train MAE: 0.039670 | Val Loss: 0.003352 | Val MAE: 0.040294
  Val Loss s-a îmbunătățit (0.003650 --> 0.003352). Se salvează modelul...
Epoca 10/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.12it/s]
Epoca 10/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.16it/s]
Epoca 11/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 10/100 | Train Loss: 0.003185 | Train MAE: 0.038134 | Val Loss: 0.003452 | Val MAE: 0.039569
Epoca 11/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.92it/s]
Epoca 11/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.21it/s]

Epoca 11/100 | Train Loss: 0.002816 | Train MAE: 0.035532 | Val Loss: 0.003272 | Val MAE: 0.038491
  Val Loss s-a îmbunătățit (0.003352 --> 0.003272). Se salvează modelul...
Epoca 12/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.06it/s]
Epoca 12/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.12it/s]

Epoca 12/100 | Train Loss: 0.002455 | Train MAE: 0.033260 | Val Loss: 0.002660 | Val MAE: 0.034442
  Val Loss s-a îmbunătățit (0.003272 --> 0.002660). Se salvează modelul...
Epoca 13/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.07it/s]
Epoca 13/100 [Val]: 100%|██████████| 129/129 [00:32<00:00,  3.97it/s]

Epoca 13/100 | Train Loss: 0.002154 | Train MAE: 0.031338 | Val Loss: 0.002640 | Val MAE: 0.032694
  Val Loss s-a îmbunătățit (0.002660 --> 0.002640). Se salvează modelul...
Epoca 14/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.80it/s]
Epoca 14/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.25it/s]

Epoca 14/100 | Train Loss: 0.001840 | Train MAE: 0.029128 | Val Loss: 0.001957 | Val MAE: 0.029002
  Val Loss s-a îmbunătățit (0.002640 --> 0.001957). Se salvează modelul...
Epoca 15/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 15/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.24it/s]

Epoca 15/100 | Train Loss: 0.001747 | Train MAE: 0.028521 | Val Loss: 0.001919 | Val MAE: 0.028650
  Val Loss s-a îmbunătățit (0.001957 --> 0.001919). Se salvează modelul...
Epoca 16/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.12it/s]
Epoca 16/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.55it/s]
Epoca 17/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 16/100 | Train Loss: 0.001525 | Train MAE: 0.027142 | Val Loss: 0.001926 | Val MAE: 0.028907
Epoca 17/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:02<00:00,  6.20it/s]
Epoca 17/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.56it/s]

Epoca 17/100 | Train Loss: 0.001454 | Train MAE: 0.026589 | Val Loss: 0.001854 | Val MAE: 0.027262
  Val Loss s-a îmbunătățit (0.001919 --> 0.001854). Se salvează modelul...
Epoca 18/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:02<00:00,  6.24it/s]
Epoca 18/100 [Val]: 100%|██████████| 129/129 [00:27<00:00,  4.62it/s]
Epoca 19/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 18/100 | Train Loss: 0.001380 | Train MAE: 0.025852 | Val Loss: 0.002303 | Val MAE: 0.032648
Epoca 19/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.13it/s]
Epoca 19/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.46it/s]

Epoca 19/100 | Train Loss: 0.001106 | Train MAE: 0.023540 | Val Loss: 0.001612 | Val MAE: 0.026835
  Val Loss s-a îmbunătățit (0.001854 --> 0.001612). Se salvează modelul...
Epoca 20/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.07it/s]
Epoca 20/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.33it/s]

Epoca 20/100 | Train Loss: 0.001040 | Train MAE: 0.022871 | Val Loss: 0.001379 | Val MAE: 0.024320
  Val Loss s-a îmbunătățit (0.001612 --> 0.001379). Se salvează modelul...
Epoca 21/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.10it/s]
Epoca 21/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.41it/s]
Epoca 22/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 21/100 | Train Loss: 0.001180 | Train MAE: 0.024328 | Val Loss: 0.001721 | Val MAE: 0.027924
Epoca 22/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.07it/s]
Epoca 22/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.52it/s]

Epoca 22/100 | Train Loss: 0.000834 | Train MAE: 0.020875 | Val Loss: 0.001395 | Val MAE: 0.024814
Epoca 23/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.06it/s]
Epoca 23/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.48it/s]

Epoca 23/100 | Train Loss: 0.000789 | Train MAE: 0.020442 | Val Loss: 0.001355 | Val MAE: 0.023776
  Val Loss s-a îmbunătățit (0.001379 --> 0.001355). Se salvează modelul...
Epoca 24/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.12it/s]
Epoca 24/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.45it/s]

Epoca 24/100 | Train Loss: 0.000767 | Train MAE: 0.020163 | Val Loss: 0.001203 | Val MAE: 0.022488
  Val Loss s-a îmbunătățit (0.001355 --> 0.001203). Se salvează modelul...
Epoca 25/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.10it/s]
Epoca 25/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.33it/s]
Epoca 26/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 25/100 | Train Loss: 0.000741 | Train MAE: 0.019851 | Val Loss: 0.001387 | Val MAE: 0.024814
Epoca 26/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.14it/s]
Epoca 26/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.47it/s]
Epoca 27/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 26/100 | Train Loss: 0.001250 | Train MAE: 0.023825 | Val Loss: 0.002172 | Val MAE: 0.032586
Epoca 27/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 27/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.44it/s]
Epoca 28/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 27/100 | Train Loss: 0.000883 | Train MAE: 0.021375 | Val Loss: 0.001532 | Val MAE: 0.026028
Epoca 28/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.12it/s]
Epoca 28/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.48it/s]

Epoca 28/100 | Train Loss: 0.000642 | Train MAE: 0.018640 | Val Loss: 0.001118 | Val MAE: 0.022160
  Val Loss s-a îmbunătățit (0.001203 --> 0.001118). Se salvează modelul...
Epoca 29/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.14it/s]
Epoca 29/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.48it/s]

Epoca 29/100 | Train Loss: 0.000552 | Train MAE: 0.017375 | Val Loss: 0.001109 | Val MAE: 0.021751
  Val Loss s-a îmbunătățit (0.001118 --> 0.001109). Se salvează modelul...
Epoca 30/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.12it/s]
Epoca 30/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.48it/s]
Epoca 31/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 30/100 | Train Loss: 0.000494 | Train MAE: 0.016488 | Val Loss: 0.001169 | Val MAE: 0.021928
Epoca 31/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.13it/s]
Epoca 31/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.38it/s]
Epoca 32/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 31/100 | Train Loss: 0.000428 | Train MAE: 0.015477 | Val Loss: 0.001235 | Val MAE: 0.022740
Epoca 32/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.13it/s]
Epoca 32/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.40it/s]
Epoca 33/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 32/100 | Train Loss: 0.000446 | Train MAE: 0.015797 | Val Loss: 0.001320 | Val MAE: 0.025329
Epoca 33/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:04<00:00,  6.06it/s]
Epoca 33/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.38it/s]
Epoca 34/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 33/100 | Train Loss: 0.000495 | Train MAE: 0.016553 | Val Loss: 0.001322 | Val MAE: 0.023148
Epoca 34/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.12it/s]
Epoca 34/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.39it/s]
Epoca 35/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 34/100 | Train Loss: 0.000499 | Train MAE: 0.016535 | Val Loss: 0.001269 | Val MAE: 0.023200
Epoca 35/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.15it/s]
Epoca 35/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.36it/s]

Epoca 35/100 | Train Loss: 0.000448 | Train MAE: 0.015788 | Val Loss: 0.001085 | Val MAE: 0.021091
  Val Loss s-a îmbunătățit (0.001109 --> 0.001085). Se salvează modelul...
Epoca 36/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:15<00:00,  5.17it/s]
Epoca 36/100 [Val]: 100%|██████████| 129/129 [00:31<00:00,  4.10it/s]

Epoca 36/100 | Train Loss: 0.000377 | Train MAE: 0.014469 | Val Loss: 0.001182 | Val MAE: 0.022352
Epoca 37/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.87it/s]
Epoca 37/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.88it/s]
Epoca 38/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 37/100 | Train Loss: 0.000364 | Train MAE: 0.014379 | Val Loss: 0.001098 | Val MAE: 0.021935
Epoca 38/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:06<00:00,  5.87it/s]
Epoca 38/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.81it/s]
Epoca 39/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 38/100 | Train Loss: 0.000339 | Train MAE: 0.013646 | Val Loss: 2.106721 | Val MAE: 0.452524
Epoca 39/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:11<00:00,  5.42it/s]
Epoca 39/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.26it/s]
Epoca 40/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 39/100 | Train Loss: 0.001354 | Train MAE: 0.024988 | Val Loss: 0.001435 | Val MAE: 0.026230
Epoca 40/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.13it/s]
Epoca 40/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.27it/s]
Epoca 41/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 40/100 | Train Loss: 0.000484 | Train MAE: 0.016357 | Val Loss: 0.001089 | Val MAE: 0.021478
Epoca 41/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.15it/s]
Epoca 41/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.28it/s]

Epoca 41/100 | Train Loss: 0.000318 | Train MAE: 0.013510 | Val Loss: 0.000996 | Val MAE: 0.020486
  Val Loss s-a îmbunătățit (0.001085 --> 0.000996). Se salvează modelul...
Epoca 42/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:05<00:00,  5.93it/s]
Epoca 42/100 [Val]: 100%|██████████| 129/129 [00:33<00:00,  3.79it/s]

Epoca 42/100 | Train Loss: 0.000264 | Train MAE: 0.012333 | Val Loss: 0.001046 | Val MAE: 0.020718
Epoca 43/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:07<00:00,  5.78it/s]
Epoca 43/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.23it/s]

Epoca 43/100 | Train Loss: 0.000234 | Train MAE: 0.011611 | Val Loss: 0.001076 | Val MAE: 0.021402
Epoca 44/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.12it/s]
Epoca 44/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.28it/s]

Epoca 44/100 | Train Loss: 0.000224 | Train MAE: 0.011425 | Val Loss: 0.001062 | Val MAE: 0.021192
Epoca 45/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.17it/s]
Epoca 45/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.27it/s]

Epoca 45/100 | Train Loss: 0.000217 | Train MAE: 0.011233 | Val Loss: 0.001092 | Val MAE: 0.021435
Epoca 46/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.17it/s]
Epoca 46/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.25it/s]

Epoca 46/100 | Train Loss: 0.000210 | Train MAE: 0.010975 | Val Loss: 0.001017 | Val MAE: 0.020568
Epoca 47/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.16it/s]
Epoca 47/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.29it/s]
Epoca 48/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 47/100 | Train Loss: 0.000211 | Train MAE: 0.011006 | Val Loss: 0.001051 | Val MAE: 0.021179
Epoca 48/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.15it/s]
Epoca 48/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.33it/s]

Epoca 48/100 | Train Loss: 0.000215 | Train MAE: 0.011042 | Val Loss: 0.001069 | Val MAE: 0.020832
Epoca 49/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.10it/s]
Epoca 49/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.23it/s]
Epoca 50/100 [Train] LR=1.0e-02:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 49/100 | Train Loss: 0.000208 | Train MAE: 0.010922 | Val Loss: 0.001088 | Val MAE: 0.021748
Epoca 50/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.10it/s]
Epoca 50/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.24it/s]

Epoca 50/100 | Train Loss: 0.000215 | Train MAE: 0.011062 | Val Loss: 0.001038 | Val MAE: 0.020737
Epoca 51/100 [Train] LR=1.0e-02: 100%|██████████| 389/389 [01:03<00:00,  6.17it/s]
Epoca 51/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.30it/s]
Epoca 52/100 [Train] LR=1.0e-03:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 51/100 | Train Loss: 0.000214 | Train MAE: 0.011018 | Val Loss: 0.001022 | Val MAE: 0.020321
  Răbdarea a expirat (după 10 epoci). Se scade Learning Rate-ul.
  Noul Learning Rate: 0.001
Epoca 52/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:04<00:00,  5.99it/s]
Epoca 52/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.30it/s]

Epoca 52/100 | Train Loss: 0.000124 | Train MAE: 0.008138 | Val Loss: 0.000952 | Val MAE: 0.019545
  Val Loss s-a îmbunătățit (0.000996 --> 0.000952). Se salvează modelul...
Epoca 53/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.08it/s]
Epoca 53/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.28it/s]

Epoca 53/100 | Train Loss: 0.000092 | Train MAE: 0.007201 | Val Loss: 0.000944 | Val MAE: 0.019515
  Val Loss s-a îmbunătățit (0.000952 --> 0.000944). Se salvează modelul...
Epoca 54/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 54/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.25it/s]

Epoca 54/100 | Train Loss: 0.000078 | Train MAE: 0.006709 | Val Loss: 0.000938 | Val MAE: 0.019415
  Val Loss s-a îmbunătățit (0.000944 --> 0.000938). Se salvează modelul...
Epoca 55/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 55/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.28it/s]

Epoca 55/100 | Train Loss: 0.000069 | Train MAE: 0.006346 | Val Loss: 0.000941 | Val MAE: 0.019451
Epoca 56/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.15it/s]
Epoca 56/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.28it/s]

Epoca 56/100 | Train Loss: 0.000063 | Train MAE: 0.006093 | Val Loss: 0.000934 | Val MAE: 0.019409
  Val Loss s-a îmbunătățit (0.000938 --> 0.000934). Se salvează modelul...
Epoca 57/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 57/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.35it/s]

Epoca 57/100 | Train Loss: 0.000058 | Train MAE: 0.005891 | Val Loss: 0.000933 | Val MAE: 0.019369
  Val Loss s-a îmbunătățit (0.000934 --> 0.000933). Se salvează modelul...
Epoca 58/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:04<00:00,  6.01it/s]
Epoca 58/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.31it/s]

Epoca 58/100 | Train Loss: 0.000054 | Train MAE: 0.005704 | Val Loss: 0.000932 | Val MAE: 0.019369
  Val Loss s-a îmbunătățit (0.000933 --> 0.000932). Se salvează modelul...
Epoca 59/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 59/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.28it/s]
Epoca 60/100 [Train] LR=1.0e-03:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 59/100 | Train Loss: 0.000052 | Train MAE: 0.005568 | Val Loss: 0.000933 | Val MAE: 0.019395
Epoca 60/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.11it/s]
Epoca 60/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.31it/s]

Epoca 60/100 | Train Loss: 0.000048 | Train MAE: 0.005385 | Val Loss: 0.000939 | Val MAE: 0.019448
Epoca 61/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.12it/s]
Epoca 61/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.17it/s]
Epoca 62/100 [Train] LR=1.0e-03:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 61/100 | Train Loss: 0.000045 | Train MAE: 0.005214 | Val Loss: 0.000935 | Val MAE: 0.019409
Epoca 62/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.09it/s]
Epoca 62/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.31it/s]
Epoca 63/100 [Train] LR=1.0e-03:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 62/100 | Train Loss: 0.000044 | Train MAE: 0.005126 | Val Loss: 0.000939 | Val MAE: 0.019435
Epoca 63/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:04<00:00,  6.02it/s]
Epoca 63/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.46it/s]

Epoca 63/100 | Train Loss: 0.000042 | Train MAE: 0.005016 | Val Loss: 0.000932 | Val MAE: 0.019361
Epoca 64/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.18it/s]
Epoca 64/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.34it/s]
Epoca 65/100 [Train] LR=1.0e-03:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 64/100 | Train Loss: 0.000041 | Train MAE: 0.004969 | Val Loss: 0.000933 | Val MAE: 0.019419
Epoca 65/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:03<00:00,  6.16it/s]
Epoca 65/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.50it/s]
Epoca 66/100 [Train] LR=1.0e-03:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 65/100 | Train Loss: 0.000038 | Train MAE: 0.004798 | Val Loss: 0.000940 | Val MAE: 0.019469
Epoca 66/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.18it/s]
Epoca 66/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.29it/s]
Epoca 67/100 [Train] LR=1.0e-03:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 66/100 | Train Loss: 0.000038 | Train MAE: 0.004763 | Val Loss: 0.000942 | Val MAE: 0.019422
Epoca 67/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.20it/s]
Epoca 67/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.37it/s]
Epoca 68/100 [Train] LR=1.0e-03:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 67/100 | Train Loss: 0.000036 | Train MAE: 0.004674 | Val Loss: 0.000938 | Val MAE: 0.019448
Epoca 68/100 [Train] LR=1.0e-03: 100%|██████████| 389/389 [01:02<00:00,  6.21it/s]
Epoca 68/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.48it/s]
Epoca 69/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 68/100 | Train Loss: 0.000035 | Train MAE: 0.004616 | Val Loss: 0.000938 | Val MAE: 0.019423
  Răbdarea a expirat (după 10 epoci). Se scade Learning Rate-ul.
  Noul Learning Rate: 0.0001
Epoca 69/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.17it/s]
Epoca 69/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.50it/s]
Epoca 70/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 69/100 | Train Loss: 0.000031 | Train MAE: 0.004274 | Val Loss: 0.000933 | Val MAE: 0.019362
Epoca 70/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.17it/s]
Epoca 70/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.46it/s]
Epoca 71/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 70/100 | Train Loss: 0.000030 | Train MAE: 0.004223 | Val Loss: 0.000935 | Val MAE: 0.019338
Epoca 71/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.15it/s]
Epoca 71/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.49it/s]
Epoca 72/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 71/100 | Train Loss: 0.000029 | Train MAE: 0.004193 | Val Loss: 0.000937 | Val MAE: 0.019361
Epoca 72/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.13it/s]
Epoca 72/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.48it/s]
Epoca 73/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 72/100 | Train Loss: 0.000029 | Train MAE: 0.004167 | Val Loss: 0.000934 | Val MAE: 0.019358
Epoca 73/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.25it/s]
Epoca 73/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.37it/s]
Epoca 74/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 73/100 | Train Loss: 0.000029 | Train MAE: 0.004160 | Val Loss: 0.000935 | Val MAE: 0.019360
Epoca 74/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.16it/s]
Epoca 74/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.43it/s]
Epoca 75/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 74/100 | Train Loss: 0.000029 | Train MAE: 0.004134 | Val Loss: 0.000936 | Val MAE: 0.019380
Epoca 75/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.18it/s]
Epoca 75/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.48it/s]
Epoca 76/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 75/100 | Train Loss: 0.000029 | Train MAE: 0.004144 | Val Loss: 0.000936 | Val MAE: 0.019399
Epoca 76/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.20it/s]
Epoca 76/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.42it/s]

Epoca 76/100 | Train Loss: 0.000029 | Train MAE: 0.004148 | Val Loss: 0.000942 | Val MAE: 0.019413
Epoca 77/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.14it/s]
Epoca 77/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.43it/s]

Epoca 77/100 | Train Loss: 0.000028 | Train MAE: 0.004110 | Val Loss: 0.000933 | Val MAE: 0.019369
Epoca 78/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.18it/s]
Epoca 78/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.42it/s]

Epoca 78/100 | Train Loss: 0.000028 | Train MAE: 0.004106 | Val Loss: 0.000937 | Val MAE: 0.019401
Epoca 79/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.19it/s]
Epoca 79/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.45it/s]
Epoca 80/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 79/100 | Train Loss: 0.000028 | Train MAE: 0.004098 | Val Loss: 0.000938 | Val MAE: 0.019378
Epoca 80/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.15it/s]
Epoca 80/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.45it/s]

Epoca 80/100 | Train Loss: 0.000028 | Train MAE: 0.004092 | Val Loss: 0.000930 | Val MAE: 0.019347
  Val Loss s-a îmbunătățit (0.000932 --> 0.000930). Se salvează modelul...
Epoca 81/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.11it/s]
Epoca 81/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.42it/s]
Epoca 82/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 81/100 | Train Loss: 0.000028 | Train MAE: 0.004085 | Val Loss: 0.000934 | Val MAE: 0.019356
Epoca 82/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.17it/s]
Epoca 82/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.45it/s]

Epoca 82/100 | Train Loss: 0.000028 | Train MAE: 0.004074 | Val Loss: 0.000938 | Val MAE: 0.019385
Epoca 83/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.16it/s]
Epoca 83/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.49it/s]

Epoca 83/100 | Train Loss: 0.000028 | Train MAE: 0.004067 | Val Loss: 0.000937 | Val MAE: 0.019377
Epoca 84/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.19it/s]
Epoca 84/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.33it/s]
Epoca 85/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 84/100 | Train Loss: 0.000027 | Train MAE: 0.004021 | Val Loss: 0.000936 | Val MAE: 0.019363
Epoca 85/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.21it/s]
Epoca 85/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.31it/s]
Epoca 86/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 85/100 | Train Loss: 0.000027 | Train MAE: 0.004034 | Val Loss: 0.000939 | Val MAE: 0.019384
Epoca 86/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.18it/s]
Epoca 86/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.53it/s]
Epoca 87/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 86/100 | Train Loss: 0.000027 | Train MAE: 0.004038 | Val Loss: 0.000941 | Val MAE: 0.019425
Epoca 87/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.15it/s]
Epoca 87/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.47it/s]

Epoca 87/100 | Train Loss: 0.000027 | Train MAE: 0.004020 | Val Loss: 0.000934 | Val MAE: 0.019340
Epoca 88/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:03<00:00,  6.17it/s]
Epoca 88/100 [Val]: 100%|██████████| 129/129 [00:30<00:00,  4.26it/s]
Epoca 89/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 88/100 | Train Loss: 0.000027 | Train MAE: 0.003984 | Val Loss: 0.000938 | Val MAE: 0.019387
Epoca 89/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.20it/s]
Epoca 89/100 [Val]: 100%|██████████| 129/129 [00:28<00:00,  4.45it/s]
Epoca 90/100 [Train] LR=1.0e-04:   0%|          | 0/389 [00:00<?, ?it/s]
Epoca 89/100 | Train Loss: 0.000027 | Train MAE: 0.003991 | Val Loss: 0.000936 | Val MAE: 0.019366
Epoca 90/100 [Train] LR=1.0e-04: 100%|██████████| 389/389 [01:02<00:00,  6.22it/s]
Epoca 90/100 [Val]: 100%|██████████| 129/129 [00:29<00:00,  4.43it/s]

Epoca 90/100 | Train Loss: 0.000027 | Train MAE: 0.004003 | Val Loss: 0.000940 | Val MAE: 0.019408
  Răbdarea a expirat (după 10 epoci). Se scade Learning Rate-ul.
  Learning Rate minim atins (0.0001). Se oprește antrenamentul.

--- Antrenament Finalizat ---
Cel mai bun model a fost salvat în: best_model_face_keypoints_zero.pth (Val Loss: 0.000930)

Process finished with exit code 0

"""
# Expresie regulată pentru a extrage datele
pattern = re.compile(
    r"Epoca (\d+)/100 \| Train Loss: ([\d.]+) \| Train MAE: ([\d.]+) \| Val Loss: ([\d.]+) \| Val MAE: ([\d.]+)"
)

# Liste pentru a stoca datele
epochs = []
train_losses = []
train_maes = []
val_losses = []
val_maes = []

# Extragem datele din log
for line in log_data.splitlines():
    match = pattern.search(line)
    if match:
        epochs.append(int(match.group(1)))
        train_losses.append(float(match.group(2)))
        train_maes.append(float(match.group(3)))
        val_losses.append(float(match.group(4)))
        val_maes.append(float(match.group(5)))

print(f"S-au extras date pentru {len(epochs)} epoci.")

# --- Plot 1: Evoluția Loss-ului (MSE) ---
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Val Loss', color='orange')
plt.title('Evoluția Loss-ului (MSE) vs. Epoci')
plt.xlabel('Epocă')
plt.ylabel('Loss (MSE) - Scală Logaritmică')
plt.yscale('log')  # Folosim scală logaritmică pentru a vedea detaliile
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig('grafic_loss.png')
print("Graficul 'grafic_loss.png' a fost salvat.")

# --- Plot 2: Evoluția "Acurateței" (MAE) ---
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_maes, label='Train MAE', color='blue')
plt.plot(epochs, val_maes, label='Val MAE', color='orange')
plt.title('Evoluția Erorii Medii Absolute (MAE) vs. Epoci')
plt.xlabel('Epocă')
plt.ylabel('Eroare Medie Absolută (MAE)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig('grafic_mae.png')
print("Graficul 'grafic_mae.png' a fost salvat.")

plt.show()
print("Graficele au fost afișate.")