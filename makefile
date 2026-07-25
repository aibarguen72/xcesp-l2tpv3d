# makefile - xcesp-l2tpv3d
# Python subproject.  Convention parallels xcesppy.

include PROJECT

.PHONY: all test clean install

all:
	@echo "$(PRJNAME) $(PRJVERSION) — Python subproject; nothing to compile."
	@echo "  make test     — run pytest suite"
	@echo "  make install  — pip install -e . (into current venv)"
	@echo "  make clean    — remove build/pycache"

test:
	python3 -m pytest -v test/

install:
	pip install -e .

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
