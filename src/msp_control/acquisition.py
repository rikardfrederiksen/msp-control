from dataclasses import dataclass

import numpy as np

from msp_control.hardware.nidaq import NIDaq
from msp_control.hardware.optoscan import OptoscanSerial, ScanConfig


@dataclass
class RawAcquisition:
    """Raw sweeps acquired during one MSP scan."""

    dark: list[np.ndarray]
    transition: np.ndarray
    data: list[np.ndarray]


class AcquisitionController:
    """Coordinate the Optoscan and NI DAQ during an MSP acquisition."""

    def __init__(
        self,
        daq: NIDaq,
        optoscan: OptoscanSerial,
    ):
        self.daq = daq
        self.optoscan = optoscan

    def acquire(self, config: ScanConfig) -> RawAcquisition:
        """Acquire one complete set of raw MSP sweeps."""

        dark_sweeps = []
        data_sweeps = []
        transition_sweep = None

        self.daq.close_shutter()
        self.daq.ir_off()

        try:
            self.optoscan.configure_scan(config)

            self.daq.configure_ai()
            self.daq.configure_timing(config.samples)
            self.daq.configure_start_trigger()

            self.daq.start()
            self.optoscan.start_scan()

            for sweep_index in range(config.cycles):

                if sweep_index == config.dark_scans:
                    self.daq.open_shutter()

                sweep = np.asarray(
                    self.daq.read_sweep(config.wavelength_points),
                    dtype=float,
                )

                if sweep_index < config.dark_scans:
                    dark_sweeps.append(sweep)
                elif sweep_index == config.dark_scans:
                    transition_sweep = sweep
                else:
                    data_sweeps.append(sweep)

            self.optoscan.wait_for_scan_complete()

        finally:
            self.daq.close_shutter()
            self.daq.ir_on()
            self.daq.close()

        return RawAcquisition(
            dark=dark_sweeps,
            transition=transition_sweep,
            data=data_sweeps,
        )
