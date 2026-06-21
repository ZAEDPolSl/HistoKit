from histokit.cohort.config import CohortConfig
from histokit.cohort.runner import CohortRunner


def main():
    config = CohortConfig.from_yaml("C:\\Repos\\HistoKit\\scripts_cohort\\configs\\cohort.yaml")

    runner = CohortRunner(config)
    runner.run()


if __name__ == "__main__":
    main()