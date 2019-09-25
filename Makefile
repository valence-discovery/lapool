PYTHON = python
PYLINT = pylint
PACKAGE_DIR = gnnpooling/
MIN_COV = 15
PYTEST = $(shell which pytest)
SHELL ?= $(shell which bash)
PIP = pip
INSTALL_SCRIPT =  ./scripts/install.sh
GIT_COMMAND := $(shell which git)
PYTEST_OPTS = --verbosity=1 --tb=short --disable-warnings
COVERAGE =  $(shell which coverage)
TAG_SCRIPT = ./scripts/tag.sh

install:
	$(PIP) install -e .

tag:
	#get highest tag number
	chmod gou+rx $(TAG_SCRIPT) && $(TAG_SCRIPT)

lint:
	-$(CONDA_PREFIX)/bin/$(PYLINT) $(PACKAGE_DIR)

clean:
	-rm -rf build
	-rm -rf htmlcov
	-rm -f .coverage
	-rm -f .noseids
	-rm -rf tests/build
	-find . -type d -name __pycache__ -exec rm -r {} \+
	-find . -iname '*.so' -exec {} \;
	-find . -iname '*.pyo' -exec {} \;
	-find . -iname '*.pyx' -exec sh -c 'echo `dirname {}`/`basename {} .pyx`.c' \; | xargs -0 -r rm -rf

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  tag            tag current commit using package version number"
	@echo "  clean          remove generated and compiled files"
	@echo "  lint           to check Python code for linting issues"
	@echo "  install    run pip install"
	@echo "  "
	@echo "You can also 'cd docs && make help' to build more documentation types"
