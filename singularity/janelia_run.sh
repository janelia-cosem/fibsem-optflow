#!/bin/bash

singularity run --nv -B $HOME/flyem-alignment,/groups/flyem $HOME/singularity/optflow.img "$1"
