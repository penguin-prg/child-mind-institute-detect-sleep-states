import os
import shutil

import wandb
import yaml


def wandb_init(CFG: dict, project: str):
    """wandbの初期化

    同名の実験がある場合は削除してから初期化する
    """

    # 古いのを捨てる
    api = wandb.Api()
    runs = api.runs(f"{project}")
    for run in runs:
        if run.config.get("exp_name") == CFG["exp_name"]:
            run.delete()

    wandb_dir = "wandb"
    if os.path.exists(wandb_dir):
        for item in os.listdir(wandb_dir):
            if CFG["exp_name"] in item:
                print(f"Deleting local run {item}")
                shutil.rmtree(os.path.join(wandb_dir, item))

    # 新しいのを作る
    wandb.init(project=project, name=CFG["exp_name"], config=CFG)
