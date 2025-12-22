import argparse
import os
import shlex
import subprocess
from pathlib import Path
from joblib import Parallel, delayed

def run_one(cmd: str, log_dir: Path, idx: int, gpus: list[int] | None):
    cmd = cmd.strip()
    if not cmd or cmd.startswith("#"):
        return (idx, "SKIP", None)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")

    # Asignación simple de GPU round-robin (1 job -> 1 GPU visible)
    if gpus:
        gpu_id = gpus[idx % len(gpus)]
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = log_dir / f"job_{idx:06d}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"$ {cmd}\n\n")
        f.flush()
        p = subprocess.run(
            cmd,
            shell=True,              # para soportar exactamente las líneas del .txt
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )

    status = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
    return (idx, status, str(log_path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("commands_file", type=str, help="Ruta a commands/*.txt")
    ap.add_argument("-j", "--n-jobs", type=int, default=1, help="Número de procesos en paralelo")
    ap.add_argument("--log-dir", type=str, default="/data/santiago.medina/deepunlearn/artifacts/logs", help="Directorio de logs")
    ap.add_argument("--gpus", type=str, default="", help="Lista de GPUs tipo '0,1,2' (opcional)")
    args = ap.parse_args()

    commands_file = Path(args.commands_file)
    log_dir = Path(args.log_dir) / commands_file.stem
    log_dir.mkdir(parents=True, exist_ok=True)

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()] or None

    lines = commands_file.read_text(encoding="utf-8").splitlines()

    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(run_one)(line, log_dir, i, gpus) for i, line in enumerate(lines)
    )

    # Resumen
    ok = sum(1 for _, s, _ in results if s == "OK")
    fail = [r for r in results if isinstance(r[1], str) and r[1].startswith("FAIL")]
    print(f"Done. OK={ok} FAIL={len(fail)} logs={log_dir}")
    if fail:
        print("Failed jobs:")
        for idx, status, lp in fail[:20]:
            print(f"  - line {idx}: {status} -> {lp}")

if __name__ == "__main__":
    main()
