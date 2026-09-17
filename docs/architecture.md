# MSP Control Software Architecture

## Purpose

This project is a replacement for the existing LabVIEW-based control
software used for microspectrophotometry (MSP).

The primary goals are:

- improve maintainability and transparency of the control software
- separate hardware control, acquisition, processing, storage, and GUI code
- retain individual raw sweeps in addition to processed spectra
- provide compatibility with the existing Python MSP analysis workflow
- use an open, documented data format
- maintain the project under Git version control
- document the instrument hardware and control protocol sufficiently that
  future changes do not require reverse-engineering the original LabVIEW code

The existing LabVIEW program is treated as the reference implementation
during development of the replacement software.


## Instrument overview

The MSP consists of several independently controlled components:

- Cairn Optoscan monochromator and controller
- Cairn PMT amplifier / photometry electronics
- data acquisition hardware
- PMT shutter
- IR illumination
- bleaching light
- microscope and associated optical hardware

The existing system uses a National Instruments DAQ.

Replacement with a LabJack or other DAQ hardware is being considered,
but has not yet been decided. The replacement hardware must reproduce
the hardware-synchronized acquisition used by the existing system.


## Hardware communication overview

Two different mechanisms are used to control and acquire data from the
instrument.

### Optoscan serial control

The Cairn Optoscan controller is configured and started over an RS-232
serial connection.

The controller exposes a Forth-based command interface. The acquisition
software sends commands that set scan parameters, compile the scan
tables, and start the scan program.

### DAQ interface

The Cairn photometry electronics and DAQ are connected by three signals:

- analog PMT data
- clock
- trigger

The DAQ also provides digital outputs used to control:

- PMT shutter
- IR illumination

The existing NI channel assignments are:

    PMT data       Dev1/ai0
    clock          /Dev1/PFI1
    trigger        /Dev1/PFI2
    shutter        Dev1/port0/line0
    IR             Dev1/port0/line1

The analog input is configured as a differential voltage input.

The acquisition uses an external sample clock and a digital-edge start
trigger. The clock and trigger originate from the Cairn photometry /
monochromator electronics.

The precise meaning and timing relationship of the CLOCK and TRIGGER
signals is still being documented.


## Acquisition architecture

The current working model of the acquisition sequence is:

    configure Optoscan over RS-232
            |
            v
    compile scan tables
            |
            v
    configure / arm DAQ
            |
            v
    send run_scan_prog to Optoscan
            |
            v
    Cairn hardware executes scan
            |
            +---- CLOCK -----> DAQ
            |
            +---- TRIGGER ---> DAQ
            |
            +---- PMT DATA --> DAQ analog input

The Cairn hardware therefore provides the real-time scan timing. The
computer does not need to generate precisely timed wavelength steps in
software.

The DAQ is configured for a finite acquisition using:

- one differential analog voltage channel
- an external sample clock
- a digital rising-edge start trigger
- a predetermined number of samples


## Measurement workflow

A measurement can be either a baseline or a specimen measurement.

Before acquisition:

1. Close the monochromator slits.
2. Configure the monochromator scan parameters.
3. Compile and set the monochromator scan tables.
4. Configure the DAQ.
5. Arm the acquisition.
6. Start the monochromator scan program.

The controller scan program includes:

1. dark sweeps
2. one transition sweep after opening the PMT shutter
3. illuminated baseline or specimen sweeps

The number of controller cycles is therefore:

    cycles = dark_sweeps + illuminated_sweeps + 1

The additional cycle is the transition sweep that is not included in
the averaged measurement.

Typical acquisition parameters are:

    dark sweeps       2
    baseline sweeps   5
    specimen sweeps   5-10

For example:

    2 dark + 1 transition + 5 baseline sweeps = 8 cycles

and:

    2 dark + 1 transition + 10 specimen sweeps = 13 cycles


## PMT shutter control

The PMT shutter is controlled through a DAQ digital output.

During acquisition the first sweeps are collected with the PMT shutter
closed and are used as dark measurements.

After the configured number of dark sweeps, the DAQ digital output opens
the shutter. The first sweep following the shutter transition is
discarded. Subsequent sweeps are used as baseline or specimen data.

The exact relationship between the Cairn TRIGGER signal, the scan
counter, and shutter switching is still being documented.


## Processing during acquisition

No individual sweeps are manually accepted or rejected during
acquisition.

All requested sweeps are acquired and retained.

For a baseline measurement:

    raw dark sweeps
            |
            v
        dark_mean

    raw baseline sweeps
            |
            v
       baseline_mean
            |
            | - dark_mean
            v
     baseline_corrected

For a specimen measurement:

    raw dark sweeps
            |
            v
        dark_mean

    raw specimen sweeps
            |
            v
       specimen_mean
            |
            | - dark_mean
            v
     specimen_corrected

Optical density is calculated relative to the selected corrected
baseline:

    OD = log10(baseline_corrected / specimen_corrected)

which is equivalent to:

    OD = -log10(specimen_corrected / baseline_corrected)

Individual sweeps can later be inspected and excluded during analysis.
Such exclusions are analysis decisions and should not alter the original
acquisition record.


## Data hierarchy

The current conceptual data hierarchy is:

    Experiment
    |
    +-- Baselines
    |   |
    |   +-- Baseline
    |   +-- Baseline
    |   +-- ...
    |
    +-- ScanGroups
        |
        +-- ScanGroup
            |
            +-- Scan
            +-- Scan
            +-- ...

Multiple baseline measurements may be acquired during an experiment.

A specimen Scan explicitly records which baseline was used for its
optical-density calculation.

The current Python data model is defined in:

    src/msp_control/data/model.py

This model is intentionally preliminary and may change as acquisition
and storage components are developed.


## Data retained during acquisition

For a baseline:

- wavelength
- individual raw dark sweeps
- individual raw baseline sweeps
- mean dark spectrum
- mean baseline spectrum
- dark-corrected baseline spectrum
- metadata

For a specimen scan:

- wavelength
- baseline identifier
- individual raw dark sweeps
- individual raw specimen sweeps
- mean dark spectrum
- mean specimen spectrum
- dark-corrected specimen spectrum
- optical density
- metadata


## File format

HDF5 is currently the preferred format for the replacement software.

The final HDF5 schema has intentionally not yet been designed. Storage
development will follow development of the acquisition and hardware
interfaces so that the file structure reflects actual instrument
requirements.

The acquisition file should preserve both raw and processed data.


## Software design principles

Hardware communication should be separated from acquisition logic.

Acquisition sequencing should be explicit rather than embedded in GUI
callbacks.

Data objects should remain simple containers. Processing, file I/O,
hardware communication, and analysis should live in separate modules.

Raw measurements should be retained whenever practical.

The acquisition file should represent what actually happened during the
experiment. Subsequent analysis decisions should not overwrite the
original acquisition record.

Hardware-specific DAQ code should be isolated behind an interface so
that acquisition logic is not tied directly to National Instruments or
another particular DAQ vendor.


## DAQ requirements for replacement hardware

A replacement for the existing NI DAQ must, at minimum, be able to:

- acquire one analog voltage input
- support differential analog input
- acquire using an external hardware sample clock
- start acquisition from a separate external digital trigger
- acquire a predetermined finite number of samples
- buffer the hardware-timed acquisition without relying on Python timing
- provide at least two digital outputs
- control the PMT shutter
- control the IR illumination

The external clock and trigger requirements should be verified carefully
before selecting replacement DAQ hardware.


## Open questions

- What exactly does one CLOCK pulse represent?
- What exactly does one TRIGGER pulse represent?
- Is TRIGGER generated once per complete scan, once per cycle, or in
  another pattern?
- How does LabVIEW count completed scans?
- At precisely what point relative to TRIGGER/CLOCK is the PMT shutter
  opened?
- What electrical levels and pulse widths are used for CLOCK and TRIGGER?
- What electrical interface is used for the PMT shutter?
- What electrical interface is used for IR control?
- What analog-output requirements are needed for bleaching-light control?
