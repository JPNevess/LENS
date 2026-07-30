"""Regenerate every figure and the results table from the committed CSVs."""
import os
import runpy
import sys

# The scripts import their helpers as top-level modules and are resolved next to
# this file, so the script runs the same from any working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

SCRIPTS = [
    "table01_results.py",
    "fig01a_competence_maps.py",
    "fig01b_lambda_internal_signals.py",
    "fig04_critical_difference.py",
    "fig05a_signal_importance.py",
    "fig05bc_robustness.py",
    "fig06_selection_effect.py",
    "fig07_signal_calibration.py",
    "fig08_selftraining_effect.py",
    "fig09_admission_calibration.py",
    "fig10ac_lambda_recovery.py",
    "fig10d_drift_detection.py",
    "fig11_referee_order.py",
]


def main():
    failed = []
    for script in SCRIPTS:
        print(f"\n== {script}")
        try:
            runpy.run_path(os.path.join(_HERE, script), run_name="__main__")
        except Exception as error:
            failed.append((script, error))
            print(f"  failed: {error}")
    print(f"\n{len(SCRIPTS) - len(failed)}/{len(SCRIPTS)} scripts completed")
    if failed:
        for script, error in failed:
            print(f"  {script}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
