from __future__ import annotations
import argparse, datetime as dt, json, platform, shutil, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def git_commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None

def git_dirty():
    try: return bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip())
    except Exception: return None

def main():
    p=argparse.ArgumentParser(); p.add_argument("--experiment",required=True); p.add_argument("--config",required=True); p.add_argument("--command",required=True); p.add_argument("--tag",default=""); a=p.parse_args()
    now=dt.datetime.now().astimezone(); suffix=f"-{a.tag}" if a.tag else ""; run_id=f"{now.strftime('%Y%m%d-%H%M%S')}-{a.experiment}{suffix}"; out=ROOT/"outputs"/run_id; out.mkdir(parents=True,exist_ok=False)
    config_path=(ROOT/a.config).resolve() if not Path(a.config).is_absolute() else Path(a.config)
    if not config_path.exists(): raise FileNotFoundError(config_path)
    shutil.copy2(config_path,out/config_path.name)
    meta={"run_id":run_id,"experiment":a.experiment,"created_at":now.isoformat(),"git_commit":git_commit(),"git_dirty":git_dirty(),"config_source":str(config_path),"config_snapshot":config_path.name,"command":a.command,"python":platform.python_version(),"platform":platform.platform()}
    (out/"run_meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps({"run_id":run_id,"output_dir":str(out.relative_to(ROOT))},indent=2))
if __name__=="__main__": main()
