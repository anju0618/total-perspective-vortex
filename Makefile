PYTHON	= uv run python
SUBJECT	= 4
RUN	= 14

.PHONY: all install visualize train predict eval lint typecheck check \
	clean fclean re help

all: help

help:
	@echo "make install    install dependencies (uv sync)"
	@echo "make visualize  plot raw and filtered EEG of subject $(SUBJECT), run $(RUN)"
	@echo "make train      train on subject $(SUBJECT), run $(RUN)"
	@echo "make predict    predict as a stream on subject $(SUBJECT), run $(RUN)"
	@echo "make eval       mean accuracy over all subjects and the 6 experiments"
	@echo "make lint       flake8"
	@echo "make typecheck  mypy"
	@echo "make check      lint + typecheck"
	@echo "make clean      remove caches and trained models"
	@echo "make fclean     clean + remove .venv"
	@echo "make re         fclean + install"
	@echo ""
	@echo "override: make train SUBJECT=1 RUN=6"

install:
	uv sync

visualize:
	$(PYTHON) src/visualize.py $(SUBJECT) $(RUN)

train:
	$(PYTHON) mybci.py $(SUBJECT) $(RUN) train

predict:
	$(PYTHON) mybci.py $(SUBJECT) $(RUN) predict

eval:
	$(PYTHON) mybci.py

lint:
	uv run flake8 mybci.py src

typecheck:
	uv run mypy mybci.py src

check: lint typecheck

clean:
	rm -rf models .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

fclean: clean
	rm -rf .venv

re: fclean install
