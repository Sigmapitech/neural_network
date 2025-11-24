##
## EPITECH PROJECT, 2025
## neural_network
## File description:
## Makefile
##

NAME = my_torch_analyzer

all: $(NAME)

$(NAME):
	echo '#!/usr/bin/env python3' > $(NAME)
	echo 'from src.main import main' >> $(NAME)
	echo 'if __name__ == "__main__": main()' >> $(NAME)
	chmod +x $(NAME)

.PHONY: all $(NAME)

clean:
	rm -rf .venv
	rm -f $(NAME)

fclean: clean
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
.PHONY: clean fclean

re: fclean all
.NOTPARALLEL: re
.PHONY: re
