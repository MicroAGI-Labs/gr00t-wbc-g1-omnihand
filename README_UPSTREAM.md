# NVIDIA foundation and upstream reference

This fork builds on NVIDIA GR00T Whole-Body Control and GEAR-SONIC. The local
operator workflow is described in the [main README](README.md) and
[data collection handbook](README_DATA_COLLECTION.md).

## Controller and model documentation

- [SONIC model card](docs/source/model_card.md)
- [Download controller and planner artifacts](docs/source/getting_started/download_models.md)
- [Native deployment installation](docs/source/getting_started/installation_deploy.md)
- [Deployment code and program flow](docs/source/references/deployment_code.md)
- [Observation configuration](docs/source/references/observation_config.md)
- [Planner model](docs/source/references/planner_onnx.md)

## Upstream projects

The [overview preserved at the pre-cleanup base](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/blob/1a4b53f2486c974177881a6a331c1f72a52a1348/README.md)
describes the original broader repository. Its training and demo commands refer
to that historical tree, not necessarily this deployment-focused checkout.

MotionBricks animation/generative-motion demos, training code, and assets have
been removed from this fork's working tree. They are independent of the current
teleop launcher. Their [original files remain in Git history](https://github.com/MicroAGI-Labs/gr00t-wbc-g1-omnihand/tree/1a4b53f2486c974177881a6a331c1f72a52a1348/motionbricks).

SONIC training and the older Decoupled WBC tree still remain pending dependency
review. In particular, current PICO code imports SONIC rotation utilities, and
the optional upper-body IK mode imports Decoupled WBC solver/model classes.

## Attribution

Retain the [citation](CITATION.cff), [license](LICENSE), and
[third-party notices](legal/) when redistributing or modifying this code.
