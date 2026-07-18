"""Generate the original five-method bar report and last-20 CSV exports."""

from baseline_suite.analysis import bar_report
from baseline_suite.config.paths import BAR_LAST20_OUTPUT_DIR


if __name__ == "__main__":
    bar_report.DEFAULT_OUTPUT_DIR = BAR_LAST20_OUTPUT_DIR
    bar_report.LAST_N_EPOCHS = 20
    bar_report.EXPORT_TRUNCATED_LAST_N_CSV = True
    bar_report.TRUNCATED_LAST_N_EPOCHS = 20
    bar_report.main()

