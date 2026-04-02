from prscope.planning.runtime.pipeline.plan_fingerprint import plan_content_fingerprint


def test_plan_content_fingerprint_stable() -> None:
    assert plan_content_fingerprint("hello") == plan_content_fingerprint("hello")
    assert plan_content_fingerprint("hello") != plan_content_fingerprint("hello ")


def test_plan_content_fingerprint_empty() -> None:
    assert plan_content_fingerprint("") == plan_content_fingerprint("")
