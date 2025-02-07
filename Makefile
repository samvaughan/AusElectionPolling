# Define the Python interpreter
PYTHON := python
PACKAGE := src

# Define the script files for each target
GET_DATA := src.scripts.get_data
FIT := src.scripts.fit_multi_normal

# Targets
default: help

.PHONY: get_data fit clean help

get_data:
	$(PYTHON) -m $(GET_DATA)

fit:
	$(PYTHON) -m $(FIT)

collect_results:
	$(PYTHON) $(COLLECT_RESULTS_SCRIPT)

clean:
	rm -rf __pycache__ *.pyc *.log

help:
	@echo "Available targets:"
	@echo "  GET_DATA       - Run the get data script"
	@echo "  fit   - Run the fit_multi_normal script"
	@echo "  clean           - Remove temporary files"
