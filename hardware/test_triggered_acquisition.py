from msp_control.acquisition import AcquisitionController
from msp_control.hardware.nidaq import NIDaq, NIDaqConfig
from msp_control.hardware.optoscan import OptoscanSerial
from msp_control.config import ScanConfig


def main():
    scan_config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=2,
        data_scans=5,
    )

    print(f"Wavelength points: {scan_config.wavelength_points}")
    print(f"Cycles: {scan_config.cycles}")
    print(f"Total samples: {scan_config.samples}")

    daq = NIDaq(NIDaqConfig())
    optoscan = OptoscanSerial("COM2")

    try:
        print("Opening Optoscan...")
        optoscan.open()
        optoscan.enter_diagnostic_mode()

        controller = AcquisitionController(
            daq=daq,
            optoscan=optoscan,
        )

        print("Starting acquisition...")
        result = controller.acquire(scan_config)

        print("Acquisition completed successfully.")

        for i, sweep in enumerate(result.dark):
            print(
                f"Dark {i}: "
                f"{len(sweep)} samples, "
                f"min={sweep.min():.4f} V, "
                f"max={sweep.max():.4f} V"
            )

        print(
            f"Transition: "
            f"{len(result.transition)} samples, "
            f"min={result.transition.min():.4f} V, "
            f"max={result.transition.max():.4f} V"
        )

        for i, sweep in enumerate(result.data):
            print(
                f"Data {i}: "
                f"{len(sweep)} samples, "
                f"min={sweep.min():.4f} V, "
                f"max={sweep.max():.4f} V"
            )

    finally:
        try:
            optoscan.exit_diagnostic_mode()
        except Exception as exc:
            print(f"Could not exit diagnostic mode: {exc}")

        optoscan.close()


if __name__ == "__main__":
    main()
