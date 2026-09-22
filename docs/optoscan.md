# Cairn Optoscan Interface

This document records the ongoing reconstruction of the Cairn Optoscan
control interface used by the existing MSP system.

The existing LabVIEW software is treated as the reference implementation.
Observed behavior is distinguished from interpretations that have not yet
been experimentally confirmed.


## Hardware

The MSP uses a Cairn Research Optoscan monochromator with a rack-mounted
controller and Cairn photometry electronics.

The Optoscan controller communicates with the acquisition computer using
RS-232.

The Cairn photometry electronics provide analog data and hardware timing
signals to the DAQ.


## Available documentation

Manufacturer documentation currently available includes:

- Cairn Modular Photometry System User Guide V2.0 (July 2001)
- Cairn Optoscan Monochromator Technical Manual 2.0 (August 2003)
- Cairn Optoscan Monochromator User Guide 2.11 (July 2010)
- Optoscan controller command dictionary (`All_CMDs.txt`)

Manufacturer documents should not necessarily be committed to the public
GitHub repository because redistribution rights have not been established.


## Serial configuration

The existing LabVIEW initialization code configures the Optoscan serial
connection as:

    baud rate          9600
    data bits          8
    parity             none
    stop bits          1
    flow control       none
    timeout            10000 ms

The VISA termination character is configured as:

    LF (0x0A)

but termination-character reading is disabled in the observed LabVIEW
initialization code.

The exact command termination sent by the LabVIEW write routine should
still be verified.


## Diagnostic interface

During startup the LabVIEW program enters the controller diagnostic
interface.

The diagnostic interface exposes a Forth-based command dictionary.

Observed initialization behavior includes entry into diagnostic mode and
a controller response of:

    ok

The precise startup byte sequence should be documented before implementing
the Python serial interface.


## Forth command interface

The LabVIEW software communicates directly with the controller's Forth
interpreter.

Examples observed in serial traffic include:

    3600 scan_start !
    7200 scan_end !
    20 scan_step_size !
    2000 scan_step_time_lo !
    0 scan_step_time_hi !
    0 scan_data>ram
    0 scan_prog# !
    0 ram>scan_data
    13 cycles !
    compile_tables
    set_scantable
    40 scan_inslit_width !
    40 scan_exslit_width !

The controller responds to these commands with:

    ok

In Forth, `!` is the store operation. For example:

    3600 scan_start !

stores the value 3600 in the controller variable `scan_start`.


## Numeric scaling

Observed LabVIEW settings and corresponding controller values show that
wavelength-related quantities are represented in units of 0.1 nm.

Examples:

    360 nm -> 3600
    720 nm -> 7200
      2 nm ->   20
      4 nm ->   40

This scaling applies to the observed:

- scan start wavelength
- scan end wavelength
- scan step size
- input slit width
- exit slit width


## Scan configuration

For an observed scan with:

    start wavelength     360 nm
    end wavelength       720 nm
    wavelength step        2 nm
    input slit width        4 nm
    exit slit width         4 nm

the configuration sequence included:

    close_slits
    3600 scan_start !
    7200 scan_end !
    20 scan_step_size !
    2000 scan_step_time_lo !
    0 scan_step_time_hi !
    0 scan_data>ram
    0 scan_prog# !
    0 ram>scan_data
    <N> cycles !
    compile_tables
    set_scantable
    40 scan_inslit_width !
    40 scan_exslit_width !

where `<N>` depends on the number of dark and illuminated sweeps.

The exact units represented by `scan_step_time_lo` and
`scan_step_time_hi` have not yet been established.

The precise roles of:

    scan_data>ram
    ram>scan_data
    scan_prog#
    compile_tables
    set_scantable

have not yet been fully documented.


## Scan execution

The LabVIEW program passes the following command to its serial-command
routine when executing a configured scan:

    run_scan_prog

`RUN_SCAN_PROG` is also present in the manufacturer-supplied controller
command dictionary.

Current interpretation:

    configure scan variables
            |
            v
      compile_tables
            |
            v
       set_scantable
            |
            v
       run_scan_prog
            |
            v
      execute scan

The observation that LabVIEW passes `run_scan_prog` to the serial-command
routine is considered established. Details of the controller's internal
implementation of this command remain undocumented.


## Scan cycles

The controller variable `cycles` includes:

- dark sweeps
- one transition sweep
- illuminated measurement sweeps

Therefore:

    cycles = dark_sweeps + illuminated_sweeps + 1

Observed examples:

    2 dark + 1 transition + 5 baseline sweeps = 8 cycles

    2 dark + 1 transition + 10 specimen sweeps = 13 cycles

This agrees with the behavior of the existing MSP acquisition software.


## Manual slit control

An observed LabVIEW sequence for opening the slits at 700 nm with
2 nm input and exit slit widths is:

    7000 scan_end !
    grating>scan_end
    20 scan_inslit_width !
    20 scan_exslit_width !
    open_slits

Current interpretation is that `grating>scan_end` moves the grating to
the wavelength stored in `scan_end`.

This interpretation is plausible from the observed behavior but should
remain marked as an interpretation until independently verified.

An observed close sequence is:

    close_slits
    5500 scan_end !

The reason for setting `scan_end` to 550 nm after closing the slits has
not yet been established.


## DAQ synchronization

The existing NI DAQ receives three signals associated with MSP
acquisition:

    PMT DATA
    CLOCK
    TRIGGER

Observed NI assignments are:

    PMT DATA     Dev1/ai0
    CLOCK        /Dev1/PFI1
    TRIGGER      /Dev1/PFI2

The analog channel is configured as a differential voltage input.

The NI acquisition is configured with:

    sample clock source     /Dev1/PFI1
    start trigger source    /Dev1/PFI2
    trigger edge            rising
    acquisition mode        finite samples

The sample-rate value supplied to DAQmx when using the external clock
appears to serve as an expected/nominal rate rather than generating the
actual sample timing. The actual acquisition timing is provided by the
external CLOCK signal.

The exact physical meaning of CLOCK and TRIGGER is still under
investigation.


## Samples per sweep

For a scan from 360 to 720 nm in 2 nm increments, the expected number of
wavelength points is:

    (720 - 360) / 2 + 1 = 181

The existing LabVIEW interface displays 181 points for this scan
configuration, consistent with this calculation.

## Serial protocol and diagnostic mode

The Optoscan controller communicates over RS-232 using the following
settings:

- 9600 baud
- 8 data bits
- no parity
- 1 stop bit
- no flow control

Normal diagnostic-mode commands are ASCII strings terminated by LF
(`\n`). The controller echoes the command and normally terminates a
successful response with `ok\r\n`.

For example:

    PC -> b"close_slits\n"
    Optoscan -> b"close_slits  ok\r\n"

### Entering diagnostic mode

The controller starts in its normal menu interface. Diagnostic mode
provides access to the Forth command interpreter used by the MSP
control software.

The following byte-level exchange was measured directly from the
controller:

    PC -> b"\n"
    Optoscan -> b"\x1b"

    PC -> b"9"
    Optoscan -> b"\x1b\x1f\r\n"

Thus, entering diagnostic mode is a two-step operation. The initial LF
wakes or synchronizes with the menu interface, and menu option `9`
enters diagnostic mode.

The complete response to `9` must be consumed before sending the first
Forth command. In particular, returning immediately after receiving
the first `ESC` byte (`0x1b`) can leave `b"\x1f\r\n"` in the serial
input buffer, causing the next command response to be parsed
incorrectly.

### Exiting diagnostic mode

Diagnostic mode is exited by sending:

    PC -> b"menu\n"

The following response was measured:

    Optoscan -> b"menu \x1b\x1b"

After this exchange the controller returns to its normal menu.

The exit response is therefore handled separately from normal Forth
commands and should not be expected to contain `ok`.


## Triggering and timing

The DAQ is configured for one finite acquisition containing all Optoscan
cycles. A single `TRIG` pulse occurs at the beginning of the complete scan
program and starts the DAQ acquisition. `CLOCK` produces one pulse per
wavelength point and serves as the external DAQ sample clock.

Thus, for \(N_\lambda\) wavelength points and \(N_c\) cycles, the total
number of acquired samples is

\[
N_{\mathrm{samples}} = N_c N_\lambda.
\]

A cycle consists of one complete wavelength sweep. The configured cycle
count is

\[
N_c = N_{\mathrm{dark}} + 1 + N_{\mathrm{data}},
\]

where the additional sweep occurs during the PMT-shutter transition and
is discarded.

## Scan step timing

`scan_step_time_lo` determines the time spent at each wavelength point.
Empirically, a value of `2000` corresponds to 2 ms and `20000` to 20 ms,
consistent with a controller time unit of 1 µs.

Normal operation uses `scan_step_time_hi = 0`. The encoding of longer
times using the high word has not yet been established.

## PMT shutter control

The PMT shutter is controlled by a DAQ digital output:

    Dev1/port0/line0

The existing acquisition code counts scan progress and opens the shutter
after the dark sweeps.

The first sweep following the shutter transition is not used as an
illuminated measurement. This accounts for the additional `+1` in the
controller cycle count.

The exact relationship between the TRIGGER signal and the scan counter
is still being investigated.


## IR control

A second DAQ digital output is assigned to IR control:

    Dev1/port0/line1

The detailed behavior and electrical interface of this control have not
yet been documented.


## Current working acquisition model

The current model is:

    PC
     |
     | RS-232 configuration
     v
    Optoscan controller
     |
     | run_scan_prog
     v
    scan execution
     |
     +-------- CLOCK --------> DAQ external sample clock
     |
     +-------- TRIGGER ------> DAQ digital start trigger
     |
     +-------- PMT DATA -----> DAQ analog input

The DAQ additionally controls:

    DAQ digital output -----> PMT shutter
    DAQ digital output -----> IR illumination

This model is based on observations of the working LabVIEW implementation.
The precise timing relationship between CLOCK, TRIGGER, wavelength
movement, PMT integration, and shutter control remains to be established.


## Questions to resolve

### Serial protocol

- What exact bytes are sent during entry into diagnostic mode?
- What command terminator is sent?
- Are commands echoed by the controller?
- Is `ok` always the normal completion response?
- Does `run_scan_prog` produce a response immediately or after scan
  completion?
- What are the units of `scan_step_time_lo` and `scan_step_time_hi`?

### Scan execution

- What exactly does `compile_tables` construct?
- What exactly does `set_scantable` select or initialize?
- What do `scan_data>ram` and `ram>scan_data` control?
- What is the role of `scan_prog#`?

### Timing

- What event generates TRIGGER?
- Is there one TRIGGER per complete acquisition or one per sweep?
- What event generates CLOCK?
- Is there exactly one CLOCK pulse per wavelength point?
- Does CLOCK occur before, during, or after the PMT integration interval?
- How does LabVIEW determine that one sweep has completed?
- How does LabVIEW count the current scan number?
- At what point relative to scan timing is the shutter opened?

### Electrical characteristics

- What are the voltage levels of CLOCK and TRIGGER?
- What are their pulse widths and polarity?
- What electrical signal controls the PMT shutter?
- What electrical signal controls IR illumination?
