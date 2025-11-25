##
## EPITECH PROJECT, 2025
## neural_network
## File description:
## Makefile
##

NAME = my_torch_analyzer

all: $(NAME)

venv:
	python -m venv venv || ./bootstrap-docker-venv.sh
	venv/bin/pip install -e .

$(NAME): venv
	ln -s venv/bin/$@ $@

.PHONY: all $(NAME)

clean:
	$(RM) $(NAME)

fclean: clean
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

mrproper: clean
	$(RM) -r venv src/*.egg-info

.PHONY: clean fclean mrproper

re: fclean all
.NOTPARALLEL: re
.PHONY: re
