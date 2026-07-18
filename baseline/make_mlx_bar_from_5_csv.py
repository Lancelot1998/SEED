"""Generate the original five-method bar report using the last 10 epochs."""

from baseline_suite.analysis import bar_report
from baseline_suite.config.paths import BAR_OUTPUT_DIR


if __name__ == "__main__":
    bar_report.DEFAULT_OUTPUT_DIR = BAR_OUTPUT_DIR
    bar_report.LAST_N_EPOCHS = 10
    bar_report.EXPORT_TRUNCATED_LAST_N_CSV = False
    bar_report.main()

