from ..segmentation.registry import SEGMENTER_REGISTRY


def build_segmenter(
    algorithm: str,
    config_path: str,
):

    try:
        item = SEGMENTER_REGISTRY[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown segmenter: {algorithm}"
        )

    config = item["config"].from_yaml(
        config_path
    )

    return item["segmenter"](config)