The Makefile discovers RTL sources directly from the `fpe/rtl` directory of the
tree selected with `RTL` (`../rtl/fpe/rtl` by default, or
`../comparative_code/<RTL>/fpe/rtl`), so no filelist is needed for Verilator.

Keep custom simulator-specific filelists here if you later move this flow to
VCS/Xcelium/Questa.
