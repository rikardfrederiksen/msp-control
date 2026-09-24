from msp_control.hardware.nidaq import NIDaq, NIDaqConfig
from msp_control.hardware.optoscan import OptoscanSerial, ScanConfig


def main():
    scan_config = ScanConfig(
        start_nm=360,
        end_nm=720,
        step_nm=2,
        step_time_ms=2,
        input_slit_nm=4,
        output_slit_nm=4,
        dark_scans=0,
        data_scans=1,
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

        print("Configuring Optoscan...")
        optoscan.configure_scan(scan_config)

        print("Configuring DAQ...")
        daq.configure_ai()
        daq.configure_timing(scan_config.samples)
        daq.configure_start_trigger()

        print("Arming DAQ...")
        daq.start()

        print("Starting Optoscan...")
        optoscan.start_scan()

        print("Reading sweeps...")

        for sweep_index in range(scan_config.cycles):
            sweep = daq.read_sweep(scan_config.wavelength_points)

            print(
                f"Sweep {sweep_index}: "
                f"{len(sweep)} samples, "
                f"min={min(sweep):.4f} V, "
                f"max={max(sweep):.4f} V"
            )

        print("Waiting for Optoscan completion...")
        optoscan.wait_for_scan_complete()

        print("Acquisition completed successfully.")

    finally:
        print("Cleaning up...")

        daq.close()

        try:
            optoscan.exit_diagnostic_mode()
        except Exception as exc:
            print(f"Could not exit diagnostic mode: {exc}")

        optoscan.close()


if __name__ == "__main__":
    main()
