"""End-to-end: train both strategies briefly, evaluate, resume, and analyze."""

import json

from retrieval import analysis, evaluate, train
from retrieval.config import load_config
from retrieval.data import read_jsonl


def _cfg(tmp_path, data_root, model_dir, strategy, steps=6):
    return load_config(None, [
        f"data.root={data_root}", f"output_dir={tmp_path / 'runs'}",
        f"model.name={model_dir}", "model.max_len=64",
        f"negatives.strategy={strategy}", "train.batch_size=4", f"train.max_steps={steps}",
        "train.warmup_steps=2", "train.num_workers=0", "train.log_every=2",
        "train.eval_every=3", "train.ckpt_every=3", "train.bf16=false", "eval.batch_size=8",
        "eval.premise_batch_size=16", "eval.n_boot=50", "data.val_limit=10",
    ])


def test_train_eval_resume_analyze(tmp_path, data_root, tiny_model_dir):
    eval_dirs = {}
    for strategy in ("in_file", "random"):
        run = train.train(_cfg(tmp_path, data_root, tiny_model_dir, strategy))
        assert (run / "best" / "dual_encoder.json").exists()
        assert (run / "last" / "trainer_state.pt").exists()
        log = list(read_jsonl(run / "log.jsonl"))
        assert any(r["event"] == "val" for r in log) and log[-1]["event"] == "done"
        rates = [r["in_file_fill_rate"] for r in log if r["event"] == "train"]
        assert all(x == 1.0 for x in rates) if strategy == "in_file" else all(x is None for x in rates)
        out = run / "eval_random_test"
        evaluate.main(["--run-dir", str(run), "--part", "test"])
        m = json.loads((out / "metrics.json").read_text())
        assert m["n_examples"] == 40 and "seen_recall@10" in m  # seen/unseen breakdown ran
        eval_dirs[strategy] = out

    # resuming continues from the saved step instead of restarting
    cfg = _cfg(tmp_path, data_root, tiny_model_dir, "in_file", steps=9)
    run = train.train(cfg, resume=True)
    log = list(read_jsonl(run / "log.jsonl"))
    resume = [r for r in log if r["event"] == "resume"][-1]
    assert resume["step"] == 6 and log[-1]["step"] == 9

    res = analysis.compare(eval_dirs["in_file"], eval_dirs["random"], n_boot=50)
    assert res["n_examples"] == 40 and "recall@10 (a-b)" in res
    rows = analysis.disagreements(eval_dirs["in_file"], eval_dirs["random"], data_root, "random")
    assert all(r["winner"] in "ab" for r in rows)
    d = analysis.did(eval_dirs["in_file"], eval_dirs["random"],
                     eval_dirs["in_file"], eval_dirs["random"], n_boot=50)
    assert abs(d["mrr"]["mean"]) < 1e-12
    assert "| random | random | test | in_file |" in analysis.table(tmp_path / "runs")
