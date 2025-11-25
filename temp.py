#!/usr/bin/env python3
import numpy as np


    # 加载数据
data = data = np.load("/workspace/OSR/cv_results/cross_validation_results_1.npy", allow_pickle=True).item()
print(data['mean_auroc'])
print(data['std_auroc'])
cv_res=data['individual_results']
for res in cv_res:
    print(f"Round {res['cv_round']}, AUROC={res['auroc']:.4f}, "
          f" unknown_classes:{res['unknown_classes']},"
          f"ID={res['num_id_samples']}, OOD={res['num_ood_samples']}")


