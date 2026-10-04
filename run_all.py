"""Run the complete training pipeline.

    python run_all.py                    # all stages
    python run_all.py --stages t1 t2     # selected stages (t1 t2 t3 t4 export)

Profile (GENAI_PROFILE=smoke|quick|full) and paths are set in src/config.py.
Tasks 2 -> 3 depend on earlier checkpoints; Task 4 is independent.
"""
import argparse
import shutil
import time

from src import config as C

STAGES = ["t1", "t2", "t3", "t4", "export"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--zip", action="store_true", help="zip outputs/ at the end (for Kaggle download)")
    a = ap.parse_args()
    print(f"profile={C.PROFILE} device={C.DEVICE} out={C.OUT} synthetic={C.SYNTHETIC}")
    times = {}
    for s in a.stages:
        t = time.time()
        print(f"\n================ {s} ================")
        if s == "t1":
            from src.tasks import task1_universal as m
        elif s == "t2":
            from src.tasks import task2_hard_routing as m
        elif s == "t3":
            from src.tasks import task3_soft_moe as m
        elif s == "t4":
            from src.tasks import task4_sketch_gan as m
        else:
            from src import export_onnx as m
        m.run()
        times[s] = round((time.time() - t) / 60, 1)
        C.save_json(times, C.out("tables", "stage_minutes.json"))
        print(f"[{s}] done in {times[s]} min")
    if a.zip:
        shutil.make_archive(str(C.OUT.parent / "outputs"), "zip", C.OUT)
        print("zipped ->", C.OUT.parent / "outputs.zip")


if __name__ == "__main__":
    main()
