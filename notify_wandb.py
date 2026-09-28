import argparse
from pathlib import Path
import wandb


def upload_and_alert(log_path: str, run_name: str = "Training_Complete_Alert"):
    run = wandb.init(
        project="AD-GBC_OpenEDS", name=run_name, job_type="notification"
    )

    file_path = Path(log_path)
    if file_path.exists():
        # 1. WandB Dashboard에 로그 파일을 Artifact로 업로드 (브라우저에서 다운로드 가능)
        artifact = wandb.Artifact("evaluation_log", type="result_log")
        artifact.add_file(str(file_path))
        run.log_artifact(artifact)

        # 로그 파일의 핵심 요약 내용 읽기 (하위 20줄)
        with open(file_path, "r") as f:
            lines = f.readlines()
        summary_text = "".join(lines[-20:])
    else:
        summary_text = "로그 파일을 찾을 수 없습니다."

    # 2. WandB 설정에 등록된 이메일 또는 슬랙으로 Alert 발송
    wandb.alert(
        title=f"작업 완료: {file_path.name}",
        text=f"훈련 및 평가가 완료되었습니다.\n\n[결과 요약]\n{summary_text}",
    )
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    args = parser.parse_args()

    upload_and_alert(args.log)
