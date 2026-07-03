from testbed.cli import dataset_videos


def test_dataset_videos_auto_camera_uses_legacy_fpv_metadata() -> None:
    images = {"fpv": object()}
    metadata = {"camera_names": "fpv"}

    camera = dataset_videos._select_camera(
        images=images,
        metadata=metadata,
        requested_camera="auto",
    )

    assert camera == "fpv"


def test_dataset_videos_auto_camera_uses_gmsl_primary_camera() -> None:
    images = {
        "video4": object(),
        "video5": object(),
        "video6": object(),
        "video7": object(),
    }
    metadata = {"camera_names": "video4,video5,video6,video7"}

    camera = dataset_videos._select_camera(
        images=images,
        metadata=metadata,
        requested_camera="auto",
    )

    assert camera == "video4"


def test_dataset_videos_explicit_camera_stays_explicit() -> None:
    images = {"video4": object()}
    metadata = {"camera_names": "video4"}

    camera = dataset_videos._select_camera(
        images=images,
        metadata=metadata,
        requested_camera="video5",
    )

    assert camera == "video5"
