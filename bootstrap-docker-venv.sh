#!/usr/bin/env bash
set -e

if [ ! -f /.dockerenv ]; then
  printf "\e[31mNOT DOCKER.\e[0m\n" >&2
  exit 1
fi

rm -rf venv

pushd pip-cache
pushd bootstrapping-pip

python3 \
  pip-25.3-py3-none-any.whl/pip \
    install                   \
    --no-index                \
    --break-system-packages   \
    --root-user-action ignore \
    \
    pip-25.3-py3-none-any.whl

popd
pushd virtualenv

pip \
    install                   \
    --no-index                \
    --break-system-packages   \
    --no-build-isolation      \
    --root-user-action ignore \
    \
    *.whl

popd
popd

virtualenv venv
venv/bin/pip \
    install  \
    --no-build-isolation \
    pip-cache/*.whl
