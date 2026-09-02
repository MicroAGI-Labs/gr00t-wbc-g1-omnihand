from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_thor_can_interfaces_are_serial_bound_and_do_not_use_mttcan_names():
    network = REPO / "configs/hosts/thor/systemd-networkd"
    right_link = (network / "20-omnihand-right.link").read_text()
    left_link = (network / "20-omnihand-left.link").read_text()
    assert "Driver=gs_usb" in right_link and "Name=can10" in right_link
    assert "2082395E534B50052" in right_link
    assert "Driver=gs_usb" in left_link and "Name=can11" in left_link
    assert "205B3973534B50042" in left_link

    for side in ("left", "right"):
        settings = (network / f"20-omnihand-{side}.network").read_text()
        assert "BitRate=1M" in settings
        assert "DataBitRate=5M" in settings
        assert "FDMode=yes" in settings


def test_sdk_installer_pins_commit_and_wheel_hash():
    installer = (REPO / "install_scripts/install_omnihand.sh").read_text()
    assert "026740d9fdd8ba32b0605fa702a992b322076f1b" in installer
    assert "3d089768492729d793c5e4b29ef23620a39fd26af27924a2f3ff78ca8f6d93ae" in installer
    assert ".venv_omnihand" in installer
