PY ?= python3

.PHONY: test model-test rtl-test demo bench wave clean

test: model-test rtl-test        ## run everything CI runs

model-test:                      ## Python cycle-model tests
	$(PY) sw/test_accel_model.py

rtl-test:                        ## Verilog testbench via Icarus Verilog
	$(PY) scripts/run_rtl_tests.py

demo:                            ## digit-recognition demo on the accelerator model
	$(PY) sw/demo.py --n 8

bench:                           ## regenerate docs/results.md and docs/speedup.png
	$(PY) sw/benchmark.py

wave:                            ## dump build/wave.vcd (open in GTKWave)
	$(PY) scripts/run_rtl_tests.py --sizes 4 --vcd

clean:
	rm -rf build __pycache__ sw/__pycache__
