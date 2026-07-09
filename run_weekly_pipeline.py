import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(command):
    print()
    print("Running:")
    print(" ".join(command))
    print("-" * 80)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default=None, help="CSV file with prompts.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to evaluate.")
    parser.add_argument("--use-case", type=str, default="custom")
    parser.add_argument("--difficulty", type=str, default="custom")
    parser.add_argument("--expected-format", type=str, default="text")
    parser.add_argument("--reference-text", type=str, default="")
    parser.add_argument("--max-tokens", type=str, default=None)
    parser.add_argument("--temperature", type=str, default="0.2")
    parser.add_argument("--judge-model", type=str, default="qwen/qwen3.7-max")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    if not args.prompt and not args.input:
        raise ValueError("Use either --prompt or --input.")

    run_id = make_run_id()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("runs") / run_id

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Weekly LLM Evaluation Pipeline")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    eval_command = [
        sys.executable,
        "run_custom_evaluation.py",
        "--output-dir",
        str(output_dir),
        "--temperature",
        str(args.temperature),
    ]

    if args.input:
        eval_command.extend(["--input", args.input])

    if args.prompt:
        eval_command.extend(["--prompt", args.prompt])
        eval_command.extend(["--use-case", args.use_case])
        eval_command.extend(["--difficulty", args.difficulty])
        eval_command.extend(["--expected-format", args.expected_format])

        if args.reference_text:
            eval_command.extend(["--reference-text", args.reference_text])

        if args.max_tokens:
            eval_command.extend(["--max-tokens", str(args.max_tokens)])

    run_command(eval_command)

    if not args.skip_judge:
        judge_command = [
            sys.executable,
            "custom_quality_judge.py",
            "--run-dir",
            str(output_dir),
            "--judge-model",
            args.judge_model,
        ]

        run_command(judge_command)
    else:
        print("Skipping judge step.")


    # Sync completed run to database.
    sync_command = [
        sys.executable,
        "sync_run_to_db.py",
        "--run-dir",
        str(output_dir),
    ]

    run_command(sync_command)

    print()
    print("=" * 80)
    print("Pipeline completed.")
    print("=" * 80)
    print(f"Run folder: {output_dir}")
    print()
    print("Open dashboard with:")
    print("python3 -m streamlit run weekly_runs_dashboard.py")
    print()
    print("Important output files:")
    print(f"- {output_dir / 'custom_model_logs.csv'}")
    print(f"- {output_dir / 'custom_summary_by_model.csv'}")
    print(f"- {output_dir / 'custom_preliminary_recommendations.csv'}")

    if not args.skip_judge:
        print(f"- {output_dir / 'custom_judge_scores.csv'}")
        print(f"- {output_dir / 'custom_final_recommendations.csv'}")
        print(f"- {output_dir / 'custom_final_summary.txt'}")


if __name__ == "__main__":
    main()
