class Exceptions:

    @staticmethod
    def invalid_metric_type(*args, **kwargs):
        raise ValueError(
            "Invalid Metric Type !"
        )
